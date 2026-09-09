# Ultralytics YOLO - streamlined object-detection tasks.py
#
# This file is intentionally limited to the modules required by the supplied
# YAML:
#   Conv, C3k2, C3k2_MKEC, SPPF, C2PSA, Concat, nn.Upsample,
#   and DTID.
#
# Minimal compatibility stubs are retained for non-detection task class names
# because other Ultralytics package files may import those symbols at startup.
# They are not functional task implementations.

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    C2PSA,
    C3k2,
    Concat,
    Conv,
    Detect,
    SPPF,
)
from ultralytics.nn.extra_modules import C3k2_MKEC, DTID
from ultralytics.nn.mm_modules.mm_parser import (
    get_multimodal_modules,
    is_multimodal_module,
    process_multimodal_layer,
)
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    colorstr,
    emojis,
    yaml_load,
)
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    get_num_params,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None


__all__ = (
    "BaseModel",
    "DetectionModel",
    "Ensemble",
    "attempt_load_one_weight",
    "attempt_load_weights",
    "guess_model_scale",
    "guess_model_task",
    "parse_model",
    "torch_safe_load",
    "yaml_model_load",
    # Compatibility names imported elsewhere by Ultralytics:
    "ClassificationModel",
    "OBBModel",
    "PoseModel",
    "RTDETRDetectionModel",
    "SegmentationModel",
    "WorldModel",
)


DETECT_CLASS = (Detect, DTID)


class BaseModel(nn.Module):
    """Base class for the streamlined detection model."""

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(
        self,
        x,
        profile=False,
        visualize=False,
        augment=False,
        embed=None,
    ):
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(
        self,
        x,
        profile=False,
        visualize=False,
        embed=None,
    ):
        # Preserve the "Dual" multimodal routing used by the first YAML layer.
        from ultralytics.nn.mm import MultiModalRouter

        mm_router = MultiModalRouter(verbose=profile)
        mm_routing_enabled, mm_input_sources = (
            mm_router.setup_multimodal_routing(x, profile)
        )

        y, dt, embeddings = [], [], []

        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )

            if profile:
                self._profile_one_layer(m, x, dt)

            if mm_routing_enabled and mm_input_sources:
                routed_x = mm_router.route_layer_input(
                    x,
                    m,
                    mm_input_sources,
                    profile,
                )
                if routed_x is not None:
                    x = routed_x

            if (
                hasattr(m, "_mm_spatial_reset")
                and m._mm_spatial_reset
            ):
                x = mm_router.reset_spatial_input(
                    x,
                    m,
                    mm_input_sources,
                    profile,
                )

            x = m(x)
            y.append(x if m.i in self.save else None)

            if visualize:
                feature_visualization(
                    x,
                    m.type,
                    m.i,
                    save_dir=visualize,
                )

            if embed and m.i in embed:
                embeddings.append(
                    nn.functional.adaptive_avg_pool2d(
                        x,
                        (1, 1),
                    )
                    .squeeze(-1)
                    .squeeze(-1)
                )
                if m.i == max(embed):
                    return torch.unbind(
                        torch.cat(embeddings, 1),
                        dim=0,
                    )

        return x

    def _predict_augment(self, x):
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not "
            "support augment=True. Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        if isinstance(x, tuple):
            x = list(x)

        c = m == self.model[-1]

        if isinstance(x, list):
            try:
                bs = x[0].size(0)
            except Exception:
                bs = x[0][0].size(0)
        else:
            bs = x.size(0)

        flops = (
            thop.profile(
                m,
                inputs=[x.copy() if c else x],
                verbose=False,
            )[0]
            / 1e9
            * 2
            / bs
            if thop
            else 0
        )

        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)

        if m == self.model[0]:
            LOGGER.info(
                f"{'time (ms)':>10s} {'GFLOPs':>10s} "
                f"{'params':>10s}  module"
            )

        LOGGER.info(
            f"{dt[-1]:10.2f} {flops:10.2f} "
            f"{get_num_params(m):10.0f}  {m.type}"
        )

        if c:
            LOGGER.info(
                f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total"
            )

    def fuse(self, verbose=True):
        """Fuse Conv+BatchNorm layers used by the supplied architecture."""
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, Conv) and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse

                if hasattr(m, "switch_to_deploy"):
                    m.switch_to_deploy()

            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        norm_types = tuple(
            value
            for key, value in nn.__dict__.items()
            if "Norm" in key
        )
        return (
            sum(
                isinstance(module, norm_types)
                for module in self.modules()
            )
            < thresh
        )

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(
            self,
            detailed=detailed,
            verbose=verbose,
            imgsz=imgsz,
        )

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]

        if isinstance(m, DETECT_CLASS):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)

        return self

    def load(self, weights, verbose=True):
        model = (
            weights["model"]
            if isinstance(weights, dict)
            else weights
        )
        state_dict = model.float().state_dict()
        state_dict = intersect_dicts(
            state_dict,
            self.state_dict(),
        )
        self.load_state_dict(state_dict, strict=False)

        if verbose:
            LOGGER.info(
                f"Transferred {len(state_dict)}/"
                f"{len(self.model.state_dict())} items "
                "from pretrained weights"
            )

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = (
            self.forward(batch["img"])
            if preds is None
            else preds
        )
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError


class DetectionModel(BaseModel):
    """Object-detection model built from the supplied YAML."""

    def __init__(
        self,
        cfg="yolo11n.yaml",
        ch=3,
        nc=None,
        verbose=True,
    ):
        super().__init__()

        self.yaml = (
            cfg
            if isinstance(cfg, dict)
            else yaml_model_load(cfg)
        )

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)

        if nc and nc != self.yaml["nc"]:
            LOGGER.info(
                f"Overriding model.yaml nc={self.yaml['nc']} "
                f"with nc={nc}"
            )
            self.yaml["nc"] = nc

        self.model, self.save = parse_model(
            deepcopy(self.yaml),
            ch=ch,
            verbose=verbose,
        )

        self.names = {
            i: f"{i}"
            for i in range(self.yaml["nc"])
        }
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = False

        # Build strides for DTID.
        m = self.model[-1]
        if isinstance(m, DETECT_CLASS):
            s = 640
            m.inplace = self.inplace

            def _forward(test_input):
                return self.forward(test_input)

            from ultralytics.nn.mm import MultiModalRouter

            mm_router = MultiModalRouter(cfg, verbose=False)
            test_ch = (
                6
                if mm_router.has_multimodal_config
                else ch
            )

            cpu_input = torch.zeros(
                2,
                test_ch,
                s,
                s,
            )

            try:
                outputs = _forward(cpu_input)
                m.stride = torch.tensor(
                    [
                        s / output.shape[-2]
                        for output in outputs
                    ]
                )
            except (RuntimeError, ValueError) as error:
                message = str(error)
                cuda_only_signatures = (
                    "Not implemented on the CPU",
                    "Input type (torch.FloatTensor) and weight type",
                    "CUDA tensor",
                    "is_cuda()",
                    "carafe_forward_impl",
                    "Pointer argument",
                )

                if (
                    any(
                        signature in message
                        for signature in cuda_only_signatures
                    )
                    and torch.cuda.is_available()
                ):
                    self.model.to(torch.device("cuda"))
                    cuda_input = cpu_input.to(
                        torch.device("cuda")
                    )
                    outputs = _forward(cuda_input)
                    m.stride = torch.tensor(
                        [
                            s / output.shape[-2]
                            for output in outputs
                        ],
                        device=cuda_input.device,
                    )
                else:
                    raise

            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.tensor([32.0])

        initialize_weights(self)

        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        img_size = x.shape[-2:]
        scales = [1, 0.83, 0.67]
        flips = [None, 3, None]
        predictions = []

        for scale, flip in zip(scales, flips):
            xi = scale_img(
                x.flip(flip) if flip else x,
                scale,
                gs=int(self.stride.max()),
            )
            yi = super().predict(xi)[0]
            yi = self._descale_pred(
                yi,
                flip,
                scale,
                img_size,
            )
            predictions.append(yi)

        predictions = self._clip_augmented(predictions)
        return torch.cat(predictions, -1), None

    @staticmethod
    def _descale_pred(
        prediction,
        flips,
        scale,
        img_size,
        dim=1,
    ):
        prediction[:, :4] /= scale
        x, y, wh, cls = prediction.split(
            (
                1,
                1,
                2,
                prediction.shape[dim] - 4,
            ),
            dim,
        )

        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x

        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, predictions):
        nl = self.model[-1].nl
        grids = sum(4**x for x in range(nl))
        excluded_layers = 1

        index = (
            predictions[0].shape[-1] // grids
        ) * sum(
            4**x
            for x in range(excluded_layers)
        )
        predictions[0] = predictions[0][..., :-index]

        index = (
            predictions[-1].shape[-1] // grids
        ) * sum(
            4 ** (nl - 1 - x)
            for x in range(excluded_layers)
        )
        predictions[-1] = predictions[-1][..., index:]

        return predictions

    def init_criterion(self):
        return v8DetectionLoss(self)


class _UnsupportedTaskModel(nn.Module):
    """Compatibility placeholder for task classes not included here."""

    task_name = "non-detection"

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            f"{self.task_name} support was removed from this "
            "streamlined tasks.py. Use DetectionModel for the "
            "supplied object-detection YAML."
        )


class SegmentationModel(_UnsupportedTaskModel):
    task_name = "segmentation"


class PoseModel(_UnsupportedTaskModel):
    task_name = "pose"


class OBBModel(_UnsupportedTaskModel):
    task_name = "oriented bounding-box detection"


class ClassificationModel(_UnsupportedTaskModel):
    task_name = "classification"


class RTDETRDetectionModel(_UnsupportedTaskModel):
    task_name = "RT-DETR"


class WorldModel(_UnsupportedTaskModel):
    task_name = "YOLO-World"


class Ensemble(nn.ModuleList):
    """Ensemble container retained for weight-loading compatibility."""

    def forward(
        self,
        x,
        augment=False,
        profile=False,
        visualize=False,
    ):
        outputs = [
            module(
                x,
                augment,
                profile,
                visualize,
            )[0]
            for module in self
        ]
        return torch.cat(outputs, 2), None


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    modules = modules or {}
    attributes = attributes or {}

    import sys
    from importlib import import_module

    try:
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(
                import_module(old_module),
                old_attr,
                getattr(
                    import_module(new_module),
                    new_attr,
                ),
            )

        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        for old in modules:
            sys.modules.pop(old, None)


class SafeClass:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
        )
        return (
            super().find_class(module, name)
            if module in safe_modules
            else SafeClass
        )


def torch_safe_load(weight, safe_only=False):
    """Load a PyTorch checkpoint with Ultralytics path compatibility."""
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)

    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence":
                    "torch.nn.Identity",
                "ultralytics.nn.tasks.YOLOv10DetectionModel":
                    "ultralytics.nn.tasks.DetectionModel",
                "ultralytics.utils.loss.v10DetectLoss":
                    "ultralytics.utils.loss.E2EDetectLoss",
            },
        ):
            if safe_only:
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = (
                    lambda file_obj:
                    SafeUnpickler(file_obj).load()
                )
                with open(file, "rb") as file_object:
                    checkpoint = torch.load(
                        file_object,
                        pickle_module=safe_pickle,
                    )
            else:
                checkpoint = torch.load(
                    file,
                    map_location="cpu",
                )

    except ModuleNotFoundError as error:
        if error.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be a YOLOv5 "
                    "checkpoint and is not forward-compatible."
                )
            ) from error

        LOGGER.warning(
            f"WARNING ⚠️ {weight} requires '{error.name}'. "
            "Auto-installing the missing requirement."
        )
        check_requirements(error.name)
        checkpoint = torch.load(
            file,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to have "
            "been saved in a nonstandard format."
        )
        checkpoint = {"model": checkpoint.model}

    return checkpoint, file


def attempt_load_weights(
    weights,
    device=None,
    inplace=True,
    fuse=False,
):
    ensemble = Ensemble()

    for weight in (
        weights
        if isinstance(weights, list)
        else [weights]
    ):
        checkpoint, weight = torch_safe_load(weight)
        args = (
            {
                **DEFAULT_CFG_DICT,
                **checkpoint["train_args"],
            }
            if "train_args" in checkpoint
            else None
        )

        model = (
            checkpoint.get("ema")
            or checkpoint["model"]
        ).to(device).float()

        model.args = args
        model.pt_path = weight
        model.task = guess_model_task(model)

        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        ensemble.append(
            model.fuse().eval()
            if fuse and hasattr(model, "fuse")
            else model.eval()
        )

    for module in ensemble.modules():
        if hasattr(module, "inplace"):
            module.inplace = inplace
        elif (
            isinstance(module, nn.Upsample)
            and not hasattr(
                module,
                "recompute_scale_factor",
            )
        ):
            module.recompute_scale_factor = None

    if len(ensemble) == 1:
        return ensemble[-1]

    LOGGER.info(f"Ensemble created with {weights}\n")

    for key in ("names", "nc", "yaml"):
        setattr(
            ensemble,
            key,
            getattr(ensemble[0], key),
        )

    ensemble.stride = ensemble[
        int(
            torch.argmax(
                torch.tensor(
                    [
                        model.stride.max()
                        for model in ensemble
                    ]
                )
            )
        )
    ].stride

    assert all(
        ensemble[0].nc == model.nc
        for model in ensemble
    )

    return ensemble


def attempt_load_one_weight(
    weight,
    device=None,
    inplace=True,
    fuse=False,
):
    checkpoint, weight = torch_safe_load(weight)

    args = {
        **DEFAULT_CFG_DICT,
        **checkpoint.get("train_args", {}),
    }

    model = (
        checkpoint.get("ema")
        or checkpoint["model"]
    ).to(device).float()

    model.args = {
        key: value
        for key, value in args.items()
        if key in DEFAULT_CFG_KEYS
    }
    model.pt_path = weight
    model.task = guess_model_task(model)

    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (
        model.fuse().eval()
        if fuse and hasattr(model, "fuse")
        else model.eval()
    )

    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = inplace
        elif (
            isinstance(module, nn.Upsample)
            and not hasattr(
                module,
                "recompute_scale_factor",
            )
        ):
            module.recompute_scale_factor = None

    return model, checkpoint


def parse_model(d, ch, verbose=True, warehouse_manager=None):
    """Parse only the layers required by the supplied paper YAML."""
    import ast as python_ast

    from ultralytics.nn.mm import MultiModalRouter

    mm_router = MultiModalRouter(d, verbose)

    max_channels = float("inf")
    nc, act, scales = (
        d.get(key)
        for key in ("nc", "activation", "scales")
    )
    depth = d.get("depth_multiple", 1.0)
    width = d.get("width_multiple", 1.0)
    scale = d.get("scale", "")

    if scales:
        if not scale:
            scale = next(iter(scales))
            LOGGER.warning(
                "WARNING ⚠️ no model scale passed. "
                f"Assuming scale='{scale}'."
            )

        scale_values = scales[scale]
        if len(scale_values) < 3:
            raise ValueError(
                f"Invalid scale definition for '{scale}': "
                f"{scale_values}"
            )

        depth, width, max_channels = scale_values[:3]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(
                f"{colorstr('activation:')} {act}"
            )

    if verbose:
        LOGGER.info(
            f"\n{'':>3}{'from':>20}{'n':>3}"
            f"{'params':>10}  {'module':<45}"
            f"{'arguments':<30}"
        )

    channels = [ch]
    layers = []
    save = []

    supported_channel_modules = {
        Conv,
        C3k2,
        C3k2_MKEC,
        SPPF,
        C2PSA,
    }
    repeat_modules = {
        C3k2,
        C3k2_MKEC,
        C2PSA,
    }

    for i, layer_config in enumerate(
        d["backbone"] + d["head"]
    ):
        if len(layer_config) < 4:
            raise ValueError(
                f"Layer {i} must contain at least "
                "[from, repeats, module, args]."
            )

        c1, mm_input_source, mm_attributes = (
            mm_router.parse_layer_config(
                layer_config,
                i,
                channels,
                verbose,
            )
        )
        f, n, module_name, args = layer_config[:4]
        args = list(args)
        original_module_name = module_name

        if (
            isinstance(module_name, str)
            and is_multimodal_module(module_name)
        ):
            multimodal_modules = get_multimodal_modules()
            module = multimodal_modules[module_name]
        elif (
            isinstance(module_name, str)
            and module_name.startswith("nn.")
        ):
            module = getattr(
                torch.nn,
                module_name[3:],
            )
        elif isinstance(module_name, str):
            try:
                module = globals()[module_name]
            except KeyError as error:
                raise KeyError(
                    f"Unsupported module '{module_name}' "
                    f"in YAML layer {i}. This streamlined "
                    "tasks.py only supports the modules used "
                    "by the supplied paper YAML."
                ) from error
        else:
            module = module_name

        for index, argument in enumerate(args):
            if isinstance(argument, str):
                with contextlib.suppress(
                    ValueError,
                    SyntaxError,
                ):
                    args[index] = (
                        locals()[argument]
                        if argument in locals()
                        else python_ast.literal_eval(argument)
                    )

        n_ = (
            max(round(n * depth), 1)
            if n > 1
            else n
        )
        n = n_

        if module in supported_channel_modules:
            if not args:
                raise ValueError(
                    f"Layer {i} module {module.__name__} "
                    "requires an output-channel argument."
                )

            if mm_input_source:
                c2 = args[0]
            else:
                c1 = channels[f]
                c2 = args[0]

            if c2 != nc:
                c2 = make_divisible(
                    min(c2, max_channels) * width,
                    8,
                )

            args = [c1, c2, *args[1:]]

            if module in repeat_modules:
                args.insert(2, n)
                n = 1

            # Preserve the official YOLO11 C3k2 behavior:
            # larger scales enable the internal C3k branch.
            if (
                module in {C3k2, C3k2_MKEC}
                and scale in "mlx"
            ):
                args[3] = True

        elif module is Concat:
            c2 = sum(channels[index] for index in f)

        elif module is nn.Upsample:
            c2 = channels[f]

        elif module is DTID:
            args.append(
                [channels[index] for index in f]
            )

            if len(args) < 2:
                raise ValueError(
                    "DTID requires [nc, hidc] "
                    "arguments in the YAML."
                )

            args[1] = make_divisible(
                min(args[1], max_channels) * width,
                8,
            )
            c2 = sum(channels[index] for index in f)

        elif module is Detect:
            args.append(
                [channels[index] for index in f]
            )
            c2 = sum(channels[index] for index in f)

        elif is_multimodal_module(
            original_module_name
        ):
            c2, args = process_multimodal_layer(
                module,
                f,
                args,
                channels,
                width,
                max_channels,
                d,
            )

        else:
            raise TypeError(
                f"Module {module} at YAML layer {i} is "
                "not required by the supplied architecture."
            )

        module_instance = (
            nn.Sequential(
                *(
                    module(*args)
                    for _ in range(n)
                )
            )
            if n > 1
            else module(*args)
        )

        module_type = (
            str(module)[8:-2]
            .replace("__main__.", "")
        )

        if mm_attributes:
            mm_router.set_module_attributes(
                module_instance,
                mm_attributes,
            )

        module_instance.np = sum(
            parameter.numel()
            for parameter in module_instance.parameters()
        )
        module_instance.i = i
        module_instance.f = f
        module_instance.type = module_type

        if verbose:
            LOGGER.info(
                f"{i:>3}{str(f):>20}{n_:>3}"
                f"{module_instance.np:10.0f}  "
                f"{module_type:<45}{str(args):<30}"
            )

        source_indices = (
            [f]
            if isinstance(f, int)
            else f
        )
        save.extend(
            index % i
            for index in source_indices
            if index != -1 and i > 0
        )

        layers.append(module_instance)

        if i == 0:
            channels = []

        channels.append(c2)

    return nn.Sequential(*layers), sorted(set(save))


def yaml_model_load(path):
    path = Path(path)

    if path.stem in (
        f"yolov{version}{scale}6"
        for scale in "nsmlx"
        for version in (5, 8)
    ):
        new_stem = re.sub(
            r"(\d+)([nslmx])6(.+)?$",
            r"\1\2-p6\3",
            path.stem,
        )
        LOGGER.warning(
            "WARNING ⚠️ Ultralytics YOLO P6 models now "
            f"use the -p6 suffix. Renaming {path.stem} "
            f"to {new_stem}."
        )
        path = path.with_name(
            new_stem + path.suffix
        )

    unified_path = re.sub(
        r"(\d+)([nslmx])(.+)?$",
        r"\1\3",
        str(path),
    )
    yaml_file = (
        check_yaml(unified_path, hard=False)
        or check_yaml(path)
    )

    data = yaml_load(yaml_file)
    data["scale"] = guess_model_scale(path)
    data["yaml_file"] = str(path)
    return data


def guess_model_scale(model_path):
    try:
        match = re.search(
            r"yolo[v]?\d+([nslmx])",
            Path(model_path).stem,
        )
        return match.group(1)
    except AttributeError:
        return ""


def guess_model_task(model):
    def cfg2task(cfg):
        module_name = str(
            cfg["head"][-1][-2]
        ).lower()
        return (
            "detect"
            if "detect" in module_name
            else None
        )

    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            task = cfg2task(model)
            if task:
                return task

    if isinstance(model, nn.Module):
        for expression in (
            "model.args",
            "model.model.args",
            "model.model.model.args",
        ):
            with contextlib.suppress(Exception):
                return eval(expression)["task"]

        for expression in (
            "model.yaml",
            "model.model.yaml",
            "model.model.model.yaml",
        ):
            with contextlib.suppress(Exception):
                task = cfg2task(eval(expression))
                if task:
                    return task

        for module in model.modules():
            if isinstance(module, DETECT_CLASS):
                return "detect"

    if isinstance(model, (str, Path)):
        path = Path(model)
        if (
            "detect" in path.parts
            or "yolo" in path.stem.lower()
        ):
            return "detect"

    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model "
        "task, assuming task='detect'."
    )
    return "detect"
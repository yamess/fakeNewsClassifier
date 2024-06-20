from argparse import ArgumentParser, Namespace
from os import mkdir, listdir
from os.path import exists, abspath, dirname
from typing import Dict, Tuple, List

from transformers import is_torch_available, is_tf_available
from transformers.pipelines import SUPPORTED_TASKS, pipeline, Pipeline
from transformers.tokenization_utils import BatchEncoding


class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """

    def __init__(self):
        super().__init__("ONNX Converter")

        self.add_argument("--model", type=str, required=True, help="Model's id or path (ex: bert-base-cased)")
        self.add_argument("--tokenizer", type=str, help="Tokenizer's id or path (ex: bert-base-cased)")
        self.add_argument("--task", type=str, default=None, choices=list(SUPPORTED_TASKS.keys()), help="Model's task")
        self.add_argument("--framework", type=str, choices=["pt", "tf"], help="Framework for loading the model")
        self.add_argument("--opset", type=int, default=-1, help="ONNX opset to use (-1 = latest)")
        self.add_argument("--check-loading", action="store_true", help="Check ONNX is able to load the model")
        self.add_argument("output")


def infer_shapes(nlp: Pipeline, framework: str) -> Tuple[List[str], List[str], Dict, BatchEncoding]:
    def build_shape_dict(tensor, is_input: bool, seq_len: int):
        axes = {0: "batch"}
        if is_input:
            if len(tensor.shape) == 2:
                axes[1] = "sequence"
            else:
                raise ValueError("Unable to infer tensor axes ({})".format(len(tensor.shape)))
        else:
            seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
            axes.update({dim: "sequence" for dim in seq_axes})

        return axes

    tokens = nlp.tokenizer.encode_plus("This is a sample output", return_tensors=framework)
    seq_len = tokens.input_ids.shape[-1]
    outputs = nlp.model(**tokens) if args.framework == "pt" else nlp.model(tokens)

    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    # Generate names
    output_names = ["output_{}".format(i) for i in range(len(outputs))]
    input_vars = list(tokens.keys())

    # Define dynamic axes
    input_dynamic_axes = {k: build_shape_dict(v, True, seq_len) for k, v in tokens.items()}
    output_dynamic_axes = {k: build_shape_dict(v, False, seq_len) for k, v in zip(output_names, outputs)}
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_vars, output_names, dynamic_axes, tokens


def load_graph_from_args(args: Namespace) -> Pipeline:
    # If no tokenizer provided
    if args.tokenizer is None:
        args.tokenizer = args.model

    print("Loading pipeline (task: {}, model: {}, tokenizer: {})".format(args.task, args.model, args.tokenizer))

    if args.opset == -1:
        from onnx.defs import onnx_opset_version

        print("Setting ONNX opset version to: {}".format(onnx_opset_version()))
        args.opset = onnx_opset_version()

    # Allocate tokenizer and model
    return pipeline(args.task, model=args.model, framework=args.framework)


def export_pytorch(nlp: Pipeline, args: Namespace):
    if not is_torch_available():
        print("Cannot export {} because PyTorch is not installed. Please install torch first.".format(args.model))
        exit(1)

    import torch
    from torch.onnx import export

    print("PyTorch: {}".format(torch.__version__))

    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, args.framework)
        tokens = tuple(tokens[key] for key in input_names)  # Need to be ordered
        export(
            nlp.model,
            tokens,
            f=args.output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            use_external_data_format=True,
            enable_onnx_checker=True,
        )


def export_tensorflow(nlp: Pipeline, args: Namespace):
    if not is_tf_available():
        print("Cannot export {} because TF is not installed. Please install torch first.".format(args.model))
        exit(1)

    print("Please note TensorFlow doesn't support exporting model > 2Gb")

    try:
        import tensorflow as tf
        from keras2onnx import convert_keras, save_model, build_io_names_tf2onnx, __version__ as k2ov

        print("TensorFlow: {}, keras2onnx: {}".format(tf.version.VERSION, k2ov))

        # Build
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, args.framework)

        # Forward
        nlp.model.predict(list(tokens.data.values()))
        onnx_model = convert_keras(nlp.model, nlp.model.name)
        save_model(onnx_model, args.output)

    except ImportError as e:
        print("Cannot import {} required to export TF model to ONNX. Please install {} first.".format(e.name, e.name))
        exit(1)


if __name__ == "__main__":
    parser = OnnxConverterArgumentParser()
    args = parser.parse_args()

    # Ensure we have an absolute path for the output
    args.output = abspath(args.output)

    # Create export folder if needed
    if exists(dirname(args.output)) and len(listdir(dirname(args.output))) > 0:
        raise ValueError("Folder {} already exists".format(args.output))
    elif not exists(dirname(args.output)):
        print("Creating folder {}".format(dirname(args.output)))
        mkdir(dirname(args.output))
    else:
        print("Folder {} already exists and is empty: {}".format(dirname(args.output), "\u2713"))

    # Load the pipeline
    nlp = load_graph_from_args(args)

    # Export the graph
    if args.framework == "pt":
        export_pytorch(nlp, args)
    else:
        export_tensorflow(nlp, args)

    if args.check_loading:
        from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
        from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

        print("Checking ONNX model loading from: {}".format(args.output))
        try:
            onnx_options = SessionOptions()
            onnx_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            session = InferenceSession(args.output, onnx_options, providers=["CPUExecutionProvider"])
            print("Model correctly loaded")
        except RuntimeException as re:
            print("Error while loading the model: {}".format(re))
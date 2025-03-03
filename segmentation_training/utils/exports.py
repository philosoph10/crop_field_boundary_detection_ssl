import torch
from pipeline.training import SegmentationModel


def export_model(model_path, export_dir, input_shape=(1, 4, 640, 640)):
    """
    Loads a trained Lightning model and exports it to JIT and ONNX formats.

    Args:
        model_path (Path): Path to the `.ckpt` model file.
        export_dir (Path): Directory where exports are saved.
        input_shape (tuple): Shape of the input tensor for ONNX export.
    """
    print(f"📦 Exporting model: {model_path.name}...")

    # ✅ Load the Lightning model
    lightning_model = SegmentationModel.load_from_checkpoint(model_path)
    lightning_model.eval()

    # ✅ Extract only the PyTorch model (detaching it from Lightning)
    model = lightning_model.model  # This is the SMP model (pure torch.nn.Module)

    export_dir.mkdir(exist_ok=True, parents=True)

    # JIT (TorchScript) Export (CPU version)
    jit_cpu_path = export_dir / model_path.with_suffix(".jit").name.replace(".jit", "_cpu.jit")
    scripted_model = torch.jit.script(model)  # ✅ No Lightning dependency
    scripted_model.save(jit_cpu_path)
    print(f"✅ JIT CPU model saved: {jit_cpu_path}")

    # JIT (TorchScript) Export (CUDA version)
    if torch.cuda.is_available():
        model_cuda = model.to("cuda")  # Move model to GPU
        jit_cuda_path = export_dir / model_path.with_suffix(".jit").name.replace(".jit", "_cuda.jit")
        scripted_model_cuda = torch.jit.script(model_cuda)
        scripted_model_cuda.save(jit_cuda_path)
        print(f"✅ JIT CUDA model saved: {jit_cuda_path}")

    # ONNX Export (CPU version)
    onnx_cpu_path = export_dir / model_path.with_suffix(".onnx").name.replace(".onnx", "_cpu.onnx")
    dummy_input_cpu = torch.randn(input_shape).to("cpu")  # Ensure input is on CPU
    torch.onnx.export(
        model.to("cpu"), dummy_input_cpu, onnx_cpu_path, opset_version=11, input_names=["input"], output_names=["output"]
    )
    print(f"✅ ONNX CPU model saved: {onnx_cpu_path}")

    # ONNX Export (CUDA version)
    if torch.cuda.is_available():
        model_cuda = model.to("cuda")  # Move model to GPU
        onnx_cuda_path = export_dir / model_path.with_suffix(".onnx").name.replace(".onnx", "_cuda.onnx")
        dummy_input_cuda = torch.randn(input_shape).to("cuda")  # Ensure input is on GPU
        torch.onnx.export(
            model_cuda, dummy_input_cuda, onnx_cuda_path, opset_version=11, input_names=["input"], output_names=["output"]
        )
        print(f"✅ ONNX CUDA model saved: {onnx_cuda_path}")

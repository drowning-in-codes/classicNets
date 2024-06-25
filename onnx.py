# Load pretrained model weights
import torch
import torchvision
import onnx
import onnxruntime

dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = torchvision.models.alexnet(pretrained=True).cuda()
input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
output_names = ["output1"]
torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names,
                  output_names=output_names)

onnx_model = onnx.load("alexnet.onnx")
onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph))
print(onnx_model.model_version)
ort_session = onnxruntime.InferenceSession("alexnet.onnx", providers=["CPUExecutionProvider"])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)

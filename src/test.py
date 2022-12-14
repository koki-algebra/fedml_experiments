import torch
import fedml

from data.UCI import load_data

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


if __name__ == "__main__":
    args = fedml.init()
    device = fedml.device.get_device(args)
    dataset, class_num = load_data(args)
    model = LogisticRegression(105, class_num)

    fedml_runner = fedml.FedMLRunner(args, device, dataset, model)
    fedml_runner.run()

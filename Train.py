import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

"""
~~~###  Класс для обучения и теста KalmanNet ###~~~
"""
class Pipeline_EKF:

    def __init__(self, folderName, modelName,filename):
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
        self.filename=filename

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch  # Number of Samples in Batch
        self.learningRate = args.lr  # Learning Rate
        self.weightDecay = args.wd  # L2 Weight Regularization - Weight Decay
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        if args.load_model:
            self.loadModel()
        self.minimal_loss=float('inf')
        self.other_filter=args.other_filter

    def saveModel(self, epoch):
        name = self.filename
        save_path = f"{self.folderName}{name}"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.minimal_loss,
        }, save_path)
        print(f"Model saved at epoch {epoch} with loss: {self.minimal_loss} ")

    def loadModel(self):
        name=self.filename
        load_path = f"{self.folderName}{name}"
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Model loaded from file {name} and epoch {epoch} with loss: {loss}")

    def init_training(self, SysModel, train_input, train_target):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.minimal_loss = float('inf')
        self.NNTrain(SysModel, train_input, train_target)  # Restart training

    def NNTrain(self, SysModel, train_input, train_target):
        self.N_E = len(train_input)

        for ti in range(0, self.N_steps):
            self.optimizer.zero_grad()
            self.model.train()
            self.model.batch_size = self.N_B
            self.model.init_hidden_KNet()

            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            y_out_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)

            n_e = random.sample(range(self.N_E), k=self.N_B)
            for ii, index in enumerate(n_e):
                y_training_batch[ii, :, :] = train_input[index]
                train_target_batch[ii, :, :] = train_target[index]
            self.model.InitSequence(SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_B, 1, 1), SysModel.T)

            for t in range(0, SysModel.T):
                x_out_training_batch[:, :, t] = self.model(torch.unsqueeze(y_training_batch[:, :, t], 2)).squeeze(2)
                y_out_training_batch[:, 0, t] = x_out_training_batch[:, 1, t]

            # Calculate the loss
            loss = self.loss_fn(y_out_training_batch, train_target_batch)

            # Check if loss is None or infinite, if so, restart the training


            loss.backward()
            print("loss on step ", ti, " : ", loss)
            if loss < self.minimal_loss:
                self.minimal_loss = loss
                self.saveModel(ti)

            self.optimizer.step()

        mse_test = nn.MSELoss(reduction='mean')(x_out_training_batch, train_target_batch)
        mse_test_dB = 10 * torch.log10(mse_test)

        print(f"MSE Test: {mse_test_dB} [dB]")

    def NNTest(self, SysModel, test_input, test_target):
        self.model.eval()
        self.model.batch_size = test_input.shape[0]
        self.model.init_hidden_KNet()

        x_out_test = torch.zeros([test_input.shape[0], SysModel.m, SysModel.T_test]).to(self.device)
        y_out_test= torch.zeros([test_input.shape[0], SysModel.n, SysModel.T_test]).to(self.device)

        with torch.no_grad():
            test_input = test_input.to(self.device)
            test_target = test_target.to(self.device)
            self.model.InitSequence(SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(test_input.shape[0], 1, 1),
                                    SysModel.T_test)

            for t in range(0, SysModel.T_test):
                x_out_test[:, :, t] = self.model(torch.unsqueeze(test_input[:, :, t], 2)).squeeze(2)
                y_out_test[:, 0, t] = x_out_test[:, 1, t]
        mse_test =self.loss_fn(y_out_test[0,0,:],test_target[0,0,:])

        # Select the first sequence from the batch for plotting
        x_out_test_np = x_out_test[0, 1, :].cpu().numpy()  # Assuming single feature output

        test_input_np=test_input[0, 0, :].cpu().numpy()

        plt.plot(x_out_test_np, label='значения KalmanNet', color='red')
        plt.plot(test_input_np, label='Наблюдения', color='black', linewidth=0.5)
        plt.title('KalmanNet при неточно заданной R')
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.legend()
        plt.show()
        return x_out_test_np, mse_test

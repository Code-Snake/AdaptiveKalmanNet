from KalmanNet_4GRU import KalmanNetNN_4GRU
from Train import Pipeline_EKF
from datetime import datetime
from options import *
from SSmodel_linear import SystemModel

def generate_data(batch_size, sequence_length, F, H, Q, R, m1_0):

    x_true = torch.zeros(batch_size, 3, sequence_length)
    y_true = torch.zeros(batch_size, 1, sequence_length)
    y_noisy = torch.zeros(batch_size, 1, sequence_length)
    x_noisy = torch.zeros(batch_size, 3, sequence_length)
    for i in range(batch_size):
        x_true[i, :, 0] = m1_0  # Истинное начальное состояние без добавления шума
        x_noisy[i, :, 0] = x_true[i, :, 0]
        y_true[i, :, 0] = H @ x_true[i, :, 0]  # Истинное наблюдение без шума
        y_noisy[i, :, 0] = H @ x_noisy[i, :, 0]+ R * torch.randn(1)  # Начальное наблюдение с шумом
        for t in range(1, sequence_length):
            x_true[i, :, t] = F @ x_true[i, :, t - 1]  # Истинное состояние без шума процесса
            x_noisy[i,:,t]= x_true[i,:,t]+ torch.matmul(Q, torch.randn(3, 1)).t()
            y_true[i, :, t] = H @ x_true[i, :, t]  # Истинное наблюдение без шума
            y_noisy[i, :, t] = H @ x_noisy[i, :, t] + R * torch.randn(1)  # Зашумленное наблюдение

    return y_true, y_noisy

today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

r2 = torch.tensor([1])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)

m2_0 = 0 * torch.eye(m)
m1_0 = torch.tensor([[0,0,1]]).float()


args.lr = 1e-4
args.wd = 1e-3
args.N_E = 100
args.N_CV = 1000

### Определяем модель ##################################################

sys_model = SystemModel(F, Q*q2, H, r2*R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)

print("Q for KalNet:",Q)
print("R for KalNet:",R)

#Генерация данных с истинной моделью
train_target_real,train_input_real = generate_data(args.N_T, args.T, F, H, Q, R, m1_0)

#Генерация данных с истинной моделью для теста
test_target,test_input = generate_data(args.N_T, args.T_test,  F, H, Q, R, m1_0)

#Генерация данных  с моделью, которая как будто нам дана
train_target,train_input = generate_data(args.N_T, args.T, F, H, Q, R, m1_0)


### Построение модели ##########################################################################################

print("KalmanNet with full model info")
KalmanNet_model = KalmanNetNN_4GRU()
KalmanNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))

## Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF("./", "KalmanNet", filename)
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)

#Запуск обучения
if (args.load_model==False):
    KalmanNet_Pipeline.NNTrain(sys_model, train_target, train_input_real)

# 5. Тестирование модели
predicted_values=KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target)

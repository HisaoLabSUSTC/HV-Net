% test_file = 'test_data_tri_2';
% test_file = 'test_data_invtri_2';
test_file = 'test_data_random_2';

HV_net_model = 'model_tri_invertri_1';
% HV_net_model = 'model_50_50_100_1';

load(['//10.20.2.245/datasets/HV-Net-datasets/result_MC_R2_', test_file, '.mat']);
load(['//10.20.2.245/datasets/HV-Net-datasets/result_', HV_net_model, '_', test_file, '_cuda.mat'])

plot(Time,Loss,'o','MarkerSize',10,'LineWidth',2);
hold on;
plot(Time1,Loss1,'o','MarkerSize',10,'LineWidth',2);
hold on;
plot(time,loss,'o','MarkerSize',10,'LineWidth',2);
set(gca,'linewidth',2,'fontsize',30);
legend('Point-based','Line-based','HV-Net');
xlabel('Time');
ylabel('Error');
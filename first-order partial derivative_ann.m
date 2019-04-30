%%%%��ջ�������
clear all
clc

%%%%5��Ӱ����������
F_name = {'GC-S_{t-1}', 'GC-S_{t-2}', 'GC-S_{t-3}', 'SXJ-F_{t-1}', 'SZ-LL_{t+1}'};

%%%%��ȡѵ�������Լ��Լ���һ��
	%%%��ȡTraining&Validation&Testing����
Data=xlsread('E:/˶ʿ����/��������5/��ҵ����/GC/����ANN/ANN/����/PMI-IVS-1d - TVT.xlsx',1,'B2:G1108');
Data=Data(:,[1:6])';

[m,n]=size(Data);

	%%%�ֱ��һ��Training&Validation��Testing����
[TV_Data_input, Setting1] = mapminmax(Data(1:m-1, 1:935), 0, 1);
[TV_Data_output, Setting2] = mapminmax(Data(m, 1:935), 0, 1);
[Te_Data_input, Setting3] = mapminmax(Data(1:m-1, 936:1107), 0, 1);
[Te_Data_output, Setting4] = mapminmax(Data(m, 936:1107), 0, 1);
Data_input=[TV_Data_input, Te_Data_input]; Data_output=[TV_Data_output, Te_Data_output];

P_train=TV_Data_input(:, 1:661); T_train=TV_Data_output(:, 1:661);
P_validation=TV_Data_input(:, 662:935); T_validation=TV_Data_output(:, 662:935);
P_test=Te_Data_input; T_test=Te_Data_output;

%%%%����ANNģ���Լ������Է���
	%%%Ǳ��������ڵ�����
hidden_nodes_no=[2,4,6,8,10];
Eval_train_Total_all=[];
Eval_validation_Total_all=[];
Eval_test_Total_all=[];

	%%%��ͬ������ڵ���Ŀ��Ӧ������ָ�꣨����һ����
Eval_train_Total2_all=[];
Eval_validation_Total2_all=[];
Eval_test_Total2_all=[];

for i=1:size(hidden_nodes_no, 2)
	%%%Ѱ�Һ��ʵ�Ȩ�ء���ֵ���ظ�������100��
	
		%%ÿһ�ض���Ŀ�����ڵ�ģ�͵ĳ�ʼ���������۾���
	Eval_train_Total=[];
	Eval_validation_Total=[];
	Eval_test_Total=[];
	Eval_train_Total2=[];
	Eval_validation_Total2=[];
	Eval_test_Total2=[];
	
	for j=1:100 %���ظ�������100��
		%%����/ѵ��BP�������Լ��������
			%��������
		Net = newff(Data_input,Data_output,hidden_nodes_no(i),{'logsig','purelin'},'trainlm');
		
			%����ѵ������
		Net.divideFcn = 'divideind';
		Net.divideParam.trainInd = 1:661;
		Net.divideParam.valInd = 662:935;
		Net.divideParam.testInd = 936:1107;
		Net.trainParam.max_fail=10;
		Net.trainParam.epochs=1000;%���ѵ������
		Net.trainParam.min_grad=0;
		Net.trainParam.mc=0.9;
		Net.trainParam.goal=0;%ѵ��Ŀ�꣺����������0
		Net.trainParam.lr=0.01;
	
			%ѵ������
		[Net, tr] = train(Net, Data_input, Data_output);
	
			%���׶���Ͻ��
		T_sim_bp_train = Net(P_train);
		T_sim_bp_validation = Net(P_validation);
		T_sim_bp_test = Net(P_test);

		%%��������
			%����һ��
		T_sim_bp_train2 = mapminmax('reverse', T_sim_bp_train, Setting2);
		T_sim_bp_validation2 = mapminmax('reverse', T_sim_bp_validation, Setting2);
		T_sim_bp_test2 = mapminmax('reverse', T_sim_bp_test, Setting4);
		
			%ѵ��������
		Eval_train=[sqrt(sum((T_sim_bp_train-T_train).^2)/size(T_train, 2)); sum(abs(T_sim_bp_train-T_train))/size(T_train, 2); ...
						1-sum((T_sim_bp_train-T_train).^2)/sum((mean(T_train)-T_train).^2)];%RMSE, MAE, NSE
						
		Eval_train2=[sqrt(sum((T_sim_bp_train2-Data(m, 1:661)).^2)/size(Data(m, 1:661), 2)); sum(abs(T_sim_bp_train2-Data(m, 1:661)))/size(T_train, 2); ...
						1-sum((T_sim_bp_train2-Data(m, 1:661)).^2)/sum((mean(Data(m, 1:661))-Data(m, 1:661)).^2)];%RMSE, MAE, NSE
		
			%��֤������
		Eval_validation=[sqrt(sum((T_sim_bp_validation-T_validation).^2)/size(T_validation, 2)); sum(abs(T_sim_bp_validation-T_validation))/size(T_validation, 2); ...
						1-sum((T_sim_bp_validation-T_validation).^2)/sum((mean(T_validation)-T_validation).^2)];%RMSE, MAE, NSE
		
		Eval_validation2=[sqrt(sum((T_sim_bp_validation2-Data(m, 662:935)).^2)/size(Data(m, 662:935), 2)); sum(abs(T_sim_bp_validation2-Data(m, 662:935)))/size(T_validation, 2); ...
						1-sum((T_sim_bp_validation2-Data(m, 662:935)).^2)/sum((mean(Data(m, 662:935))-Data(m, 662:935)).^2)];%RMSE, MAE, NSE
			
			%����������
		Eval_test=[sqrt(sum((T_sim_bp_test-T_test).^2)/size(T_test, 2)); sum(abs(T_sim_bp_test-T_test))/size(T_test, 2); ...
						1-sum((T_sim_bp_test-T_test).^2)/sum((mean(T_test)-T_test).^2)];%RMSE, MAE, NSE
						
		Eval_test2=[sqrt(sum((T_sim_bp_test2-Data(m, 936:1107)).^2)/size(Data(m, 936:1107), 2)); sum(abs(T_sim_bp_test2-Data(m, 936:1107)))/size(T_test, 2); ...
						1-sum((T_sim_bp_test2-Data(m, 936:1107)).^2)/sum((mean(Data(m, 936:1107))-Data(m, 936:1107)).^2)];%RMSE, MAE, NSE
			
		%%����ѵ��������
		save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\��ͬ��ʼ����ѵ��ģ��\\my_bp', num2str(j), '.mat'),'Net')
		
		%%�õ�ѵ�������۽�������������۽��
		Eval_train_Total=[Eval_train_Total Eval_train];
		Eval_validation_Total=[Eval_validation_Total Eval_validation];
		Eval_test_Total=[Eval_test_Total Eval_test];
		Eval_train_Total2=[Eval_train_Total2 Eval_train2];
		Eval_validation_Total2=[Eval_validation_Total2 Eval_validation2];
		Eval_test_Total2=[Eval_test_Total2 Eval_test2];

	end
	
	%%%����ѡ����Ȩ�ء���ֵ���������Է����Լ�����ͼ��
		%%ȷ������ṹ�Լ����浽�ض����ļ���
	Net_no=find(Eval_test_Total2(1, :)==min(min(Eval_test_Total2(1, :))));%��RMSE��Ϊѡ������
	load(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\��ͬ��ʼ����ѵ��ģ��\\my_bp', num2str(Net_no), '.mat'))%����ͬ������
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\ѡ�е�ģ��\\my_bp_HNN',num2str(hidden_nodes_no(i)),'_',num2str(Net_no),'.mat'),'Net')
	
		%%ÿһ�������ڵ�����Ӧģ�͵�����
	Eval_train_Total_all=[Eval_train_Total_all Eval_train_Total(:, Net_no)];
	Eval_validation_Total_all=[Eval_validation_Total_all Eval_validation_Total(:, Net_no)];
	Eval_test_Total_all=[Eval_test_Total_all Eval_test_Total(:, Net_no)];	
	
	Eval_train_Total2_all=[Eval_train_Total2_all Eval_train_Total2(:, Net_no)];
	Eval_validation_Total2_all=[Eval_validation_Total2_all Eval_validation_Total2(:, Net_no)];
	Eval_test_Total2_all=[Eval_test_Total2_all Eval_test_Total2(:, Net_no)];
	
		%%������Ͻ��ͼ��
	X_Scale=0:8000; Y_Scale=X_Scale;
	figure(i);
	sz = 6;
	MarkerFaceColor='k';
	subplot(3,3,1);scatter(Data(m, 1:661), mapminmax('reverse', Net(P_train), Setting2), sz, MarkerFaceColor); box on;
	hold on; plot(Y_Scale,'--');
	subplot(3,3,2);scatter(Data(m, 662:935), mapminmax('reverse', Net(P_validation), Setting2), sz, MarkerFaceColor); box on;
	hold on; plot(Y_Scale,'--');
	subplot(3,3,3);scatter(Data(m, 936:1107), mapminmax('reverse', Net(P_test), Setting4), sz, MarkerFaceColor); box on;
	hold on; plot(Y_Scale,'--');
	subplot(3,3,[4 5 6]); plot(1:661,Data(m, 1:661),'color',[.5 .5 .5],'linewidth',1.5); box on;
	hold on; scatter(1:661, mapminmax('reverse', Net(P_train), Setting2),'b.');
	subplot(3,3,[7 8 9]); plot(1:446,Data(m, 662:1107),'color',[.5 .5 .5],'linewidth',1.5); box on;
	hold on; scatter(1:446,[mapminmax('reverse', Net(P_validation), Setting2), mapminmax('reverse', Net(P_test), Setting4)],'b.');
	
		%%������������ӵ����һ��ƫ�������������Լ���λ��
	w1=Net.iw{1,1};
	theta1=Net.b{1};
	w2=Net.lw{2,1};
	
			%ѵ���ڼ���ά����洢
	Input_SA_train=[];
	SSD_train=[];
	Five_percentiles_train=[];
	for k=1: m-1
		Accumulation=0;
		for l=1: hidden_nodes_no(i)
			Accumulation=Accumulation+(power((1+exp(-((w1(l, :)*P_train(1:m-1, :))+theta1(l)))),(-1))).*...
									(1-(power((1+exp(-((w1(l, :)*P_train(1:m-1, :))+theta1(l)))),(-1))))*(w1(l, k)*w2(l));
		end
		Input_SA_train=[Input_SA_train; ((1-Net(P_train)).*P_train(k, :)).*Accumulation];
		SSD_train=[SSD_train; [sum((Input_SA_train(k, :)).^2)]];
		Sort_SA=[sort(Input_SA_train(k, :))];
		Five_percentiles_train = [Five_percentiles_train, [Sort_SA(round(length(Input_SA_train(k, :))*0.10));...
									Sort_SA(round(length(Input_SA_train(k, :))*0.25));Sort_SA(round(length(Input_SA_train(k, :))*0.50));...
										Sort_SA(round(length(Input_SA_train(k, :))*0.75));Sort_SA(round(length(Input_SA_train(k, :))*0.90))]];%    10%, 25%, 50%, 75%, 90%
	end
	Input_SA_train_all(:, :, i) = Input_SA_train;
	SSD_train_all(:, :, i) = SSD_train;
	Five_percentiles_train_all(:, :, i) = Five_percentiles_train;
	
			%��֤�ڼ���ά����洢
	Input_SA_validation=[];
	SSD_validation=[];
	Five_percentiles_validation=[];
	for k=1: m-1
		Accumulation=0;
		for l=1: hidden_nodes_no(i)
			Accumulation=Accumulation+(power((1+exp(-((w1(l, :)*P_validation(1:m-1, :))+theta1(l)))),(-1))).*...
									(1-(power((1+exp(-((w1(l, :)*P_validation(1:m-1, :))+theta1(l)))),(-1))))*(w1(l, k)*w2(l));
		end
		Input_SA_validation=[Input_SA_validation; ((1-Net(P_validation)).*P_validation(k, :)).*Accumulation];
		SSD_validation=[SSD_validation; [sum((Input_SA_validation(k, :)).^2)]];
		Sort_SA=[sort(Input_SA_validation(k, :))];
		Five_percentiles_validation = [Five_percentiles_validation, [Sort_SA(round(length(Input_SA_validation(k, :))*0.10));...
									Sort_SA(round(length(Input_SA_validation(k, :))*0.25));Sort_SA(round(length(Input_SA_validation(k, :))*0.50));...
										Sort_SA(round(length(Input_SA_validation(k, :))*0.75));Sort_SA(round(length(Input_SA_validation(k, :))*0.90))]];%    10%, 25%, 50%, 75%, 90%
	end
	Input_SA_validation_all(:, :, i) = Input_SA_validation;
	SSD_validation_all(:, :, i) = SSD_validation;
	Five_percentiles_validation_all(:, :, i) = Five_percentiles_validation;
	
			%�����ڼ���ά����洢
	Input_SA_test=[];
	SSD_test=[];
	Five_percentiles_test=[];
	for k=1: m-1
		Accumulation=0;
		for l=1: hidden_nodes_no(i)
			Accumulation=Accumulation+(power((1+exp(-((w1(l, :)*P_test(1:m-1, :))+theta1(l)))),(-1))).*...
									(1-(power((1+exp(-((w1(l, :)*P_test(1:m-1, :))+theta1(l)))),(-1))))*(w1(l, k)*w2(l));
		end
		Input_SA_test=[Input_SA_test; ((1-Net(P_test)).*P_test(k, :)).*Accumulation];
		SSD_test=[SSD_test; [sum((Input_SA_test(k, :)).^2)]];
		Sort_SA=[sort(Input_SA_test(k, :))];
		Five_percentiles_test = [Five_percentiles_test, [Sort_SA(round(length(Input_SA_test(k, :))*0.10));...
									Sort_SA(round(length(Input_SA_test(k, :))*0.25));Sort_SA(round(length(Input_SA_test(k, :))*0.50));...
										Sort_SA(round(length(Input_SA_test(k, :))*0.75));Sort_SA(round(length(Input_SA_test(k, :))*0.90))]];%    10%, 25%, 50%, 75%, 90%

	end
	Input_SA_test_all(:, :, i) = Input_SA_test;
	SSD_test_all(:, :, i) = SSD_test;
	Five_percentiles_test_all(:, :, i) = Five_percentiles_test;
		
		%%���������ԡ������Ժͷ�λ�����
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Input_SA_train_HNN',num2str(hidden_nodes_no(i)),'.mat'),'Input_SA_train')
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Input_SA_validation_HNN',num2str(hidden_nodes_no(i)),'.mat'),'Input_SA_validation')
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Input_SA_test_HNN',num2str(hidden_nodes_no(i)),'.mat'),'Input_SA_test')

	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\SSD_train_HNN',num2str(hidden_nodes_no(i)),'.mat'),'SSD_train')
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\SSD_validation_HNN',num2str(hidden_nodes_no(i)),'.mat'),'SSD_validation')
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\SSD_test_HNN',num2str(hidden_nodes_no(i)),'.mat'),'SSD_test')
	
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Percentiles_train_HNN',num2str(hidden_nodes_no(i)),'.mat'),'Five_percentiles_train')
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Percentiles_validation_HNN',num2str(hidden_nodes_no(i)),'.mat'),'Five_percentiles_validation')
	save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Percentiles_test_HNN',num2str(hidden_nodes_no(i)),'.mat'),'Five_percentiles_test')

end

	%%%������������ڵ��Ӧ���۽��
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���۽��\\Eval_train_Total_all','.mat'),'Eval_train_Total_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���۽��\\Eval_validation_Total_all','.mat'),'Eval_validation_Total_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���۽��\\Eval_test_Total_all','.mat'),'Eval_test_Total_all')

save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���۽��\\Eval_train_Total2_all','.mat'),'Eval_train_Total2_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���۽��\\Eval_validation_Total2_all','.mat'),'Eval_validation_Total2_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���۽��\\Eval_test_Total2_all','.mat'),'Eval_test_Total2_all')


	%%%���������ԡ������Ժͷ�λ�������ά����
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Input_SA_train_all','.mat'),'Input_SA_train_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Input_SA_validation_all','.mat'),'Input_SA_validation_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Input_SA_test_all','.mat'),'Input_SA_test_all')

save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\SSD_train_all','.mat'),'SSD_train_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\SSD_validation_all','.mat'),'SSD_validation_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\SSD_test_all','.mat'),'SSD_test_all')

save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Percentiles_train_all','.mat'),'Five_percentiles_train_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Percentiles_validation_all','.mat'),'Five_percentiles_validation_all')
save(strcat('E:\\˶ʿ����\\��������5\\��ҵ����\\GC\\����ANN\\ANN\\���м�����\\Percentiles_test_all','.mat'),'Five_percentiles_test_all')
	
	%%%���������Է���ͼ��
		%%ѵ����
figure(i+10)
sz = 5;
MarkerFaceColor='k';
for x = 1: size(hidden_nodes_no, 2)
	for y = 1: (m-1)
		ax1 = subplot(size(hidden_nodes_no, 2), m-1, (x-1)*(m-1)+y); scatter(Data(m, 1:661), Input_SA_train_all(y,:,x), sz, MarkerFaceColor, 'filled')
		set(gca,'ygrid','on'); axis([ax1],[0, 8000, -0.6, 0.7]); box on; 
		set(gca,'XMinorTick','on');set(gca,'YMinorTick','on');
		set(gca,'linewidth',1.0,'FontName','Arial','FontSize',8,'FontWeight','bold');
		set(gca,'FontSize',5.0,'Fontname', 'Times New Roman');
		set(gcf,'paperpositionmode','auto');
		title(F_name(y),'FontName','Arial','FontSize',8,'FontWeight','bold'); 
		xlabel('Salinity(mg/L)','FontName','Arial','FontSize',7,'FontWeight','bold'); ylabel('Relative Sensitivity','FontName','Arial','FontSize',7,'FontWeight','bold');
	end
end
		%%��֤��
figure(i+11)
sz = 5;
MarkerFaceColor='k';
for x = 1: size(hidden_nodes_no, 2)
	for y = 1: (m-1)
		ax2 = subplot(size(hidden_nodes_no, 2), m-1, (x-1)*(m-1)+y); scatter(Data(m, 662:935), Input_SA_validation_all(y,:,x), sz, MarkerFaceColor, 'filled')
		set(gca,'ygrid','on');axis([ax2],[0, 8000, -0.6, 0.7]);box on;
		set(gca,'XMinorTick','on');set(gca,'YMinorTick','on');
		set(gca,'linewidth',1.0,'FontName','Arial','FontSize',8,'FontWeight','bold');
		set(gca,'FontSize',5.0,'Fontname', 'Times New Roman');
		set(gcf,'paperpositionmode','auto');
		title(F_name(y),'FontName','Arial','FontSize',8,'FontWeight','bold'); 
		xlabel('Salinity(mg/L)','FontName','Arial','FontSize',7,'FontWeight','bold'); ylabel('Relative Sensitivity','FontName','Arial','FontSize',7,'FontWeight','bold');
	end
end
		%%������
figure(i+12)
sz = 5;
MarkerFaceColor='k';
for x = 1: size(hidden_nodes_no, 2)
	for y = 1: (m-1)
		ax3 = subplot(size(hidden_nodes_no, 2), m-1, (x-1)*(m-1)+y); scatter(Data(m, 936:1107), Input_SA_test_all(y,:,x), sz, MarkerFaceColor, 'filled')
		set(gca,'ygrid','on');axis([ax3],[0, 8000, -0.6, 0.7]);box on;
		set(gca,'XMinorTick','on');set(gca,'YMinorTick','on');
		set(gca,'linewidth',1.0,'FontName','Arial','FontSize',8,'FontWeight','bold');
		set(gca,'FontSize',5.0,'Fontname', 'Times New Roman');
		set(gcf,'paperpositionmode','auto');
		title(F_name(y),'FontName','Arial','FontSize',8,'FontWeight','bold'); 
		xlabel('Salinity(mg/L)','FontName','Arial','FontSize',7,'FontWeight','bold'); ylabel('Relative Sensitivity','FontName','Arial','FontSize',7,'FontWeight','bold');	
	end
end
		%%ѵ����+��֤��+������
%figure(i+13)
%sz = 5;
%MarkerFaceColor='k';
%for x = 1: size(hidden_nodes_no, 2)
%	for y = 1: (m-1)
%		ax3 = subplot(size(hidden_nodes_no, 2), m-1, (x-1)*(m-1)+y); scatter(Data(m, 1:1107), [Input_SA_train_all(y,:,x),Input_SA_validation_all(y,:,x),Input_SA_test_all(y,:,x),], sz, MarkerFaceColor, 'filled')
%		axis([ax3],[0, 8000, -0.6, 0.7]);box on;
%	end
%end

	%%%���ƹ����ԡ���λ������ͼ��
for i=1:size(hidden_nodes_no, 2)
	figure(i+100)
		%%ѵ����+��֤��+������
	sz = 100;
	yrange=[5,4,3,2,1];
	subplot(2,6,[1 3]);hold on; box on;
	scatter(Five_percentiles_train_all(1, :, i), yrange, sz, [.5 .5 .5], 's', 'fill');  % 10%
	scatter(Five_percentiles_train_all(2, :, i), yrange, sz, 'g', '^', 'fill');  % 25%
	scatter(Five_percentiles_train_all(3, :, i), yrange, sz, 'r', 'h', 'fill');  % 50%
	scatter(Five_percentiles_train_all(4, :, i), yrange, sz, 'y', 'd', 'fill');  % 75%
	scatter(Five_percentiles_train_all(5, :, i), yrange, sz, 'm', 'o', 'fill');  % 90%
	xlim([-inf inf]);ylim([0 6]);
	legend('D10','D25','D50','D75','D90','Location','SouthEast');
	set(gca,'ytick',[1:5]);
	set(gca,'yticklabel',['Fr5';'Fr4';'Fr3';'Fr2';'Fr1']);
	set(gca,'ygrid','on');set(gca,'xgrid','on');
	set(gca,'XMinorTick','on');set(gca,'YMinorTick','on');
	set(gca,'linewidth',1.5,'FontName','Times New Roman','FontSize',10,'FontWeight','bold');
	title('Training Period','FontName','Arial','FontSize',12,'FontWeight','bold'); 
	xlabel('Relative Sensitivity','FontName','Arial','FontSize',12,'FontWeight','bold');
	
	subplot(2,6,[4 6]);hold on;box on;
	scatter(Five_percentiles_validation_all(1, :, i), yrange, sz, [.5 .5 .5], 's', 'fill');  % 10%
	scatter(Five_percentiles_validation_all(2, :, i), yrange, sz, 'g', '^', 'fill'); % 25%
	scatter(Five_percentiles_validation_all(3, :, i), yrange, sz, 'r', 'h', 'fill');  % 50%
	scatter(Five_percentiles_validation_all(4, :, i), yrange, sz, 'y', 'd', 'fill');  % 75%
	scatter(Five_percentiles_validation_all(5, :, i), yrange, sz, 'm', 'o', 'fill'); % 90%
	xlim([-inf inf]);ylim([0 6]);
	legend('D10','D25','D50','D75','D90','Location','SouthEast');
	set(gca,'ytick',[1:5]);
	set(gca,'yticklabel',['Fr5';'Fr4';'Fr3';'Fr2';'Fr1']);
	set(gca,'ygrid','on');set(gca,'xgrid','on');
	set(gca,'XMinorTick','on');set(gca,'YMinorTick','on');
	set(gca,'linewidth',1.5,'FontName','Times New Roman','FontSize',10,'FontWeight','bold');
	title('Validation Period','FontName','Arial','FontSize',12,'FontWeight','bold'); 
	xlabel('Relative Sensitivity','FontName','Arial','FontSize',12,'FontWeight','bold');
	
	subplot(2,6,[7 9]);hold on;box on;
	scatter(Five_percentiles_test_all(1, :, i), yrange, sz, [.5 .5 .5], 's', 'fill');  % 10%
	scatter(Five_percentiles_test_all(2, :, i), yrange, sz, 'g', '^', 'fill');  % 25%
	scatter(Five_percentiles_test_all(3, :, i), yrange, sz, 'r', 'h', 'fill');  % 50%
	scatter(Five_percentiles_test_all(4, :, i), yrange, sz, 'y', 'd', 'fill');  % 75%
	scatter(Five_percentiles_test_all(5, :, i), yrange, sz, 'm', 'o', 'fill'); % 90%
	xlim([-inf inf]);ylim([0 6]);
	legend('D10','D25','D50','D75','D90','Location','SouthEast');
	set(gca,'ytick',[1:5]);
	set(gca,'yticklabel',['Fr5';'Fr4';'Fr3';'Fr2';'Fr1']);
	set(gca,'ygrid','on');set(gca,'xgrid','on');
	set(gca,'XMinorTick','on');set(gca,'YMinorTick','on');
	set(gca,'linewidth',1.5,'FontName','Times New Roman','FontSize',10,'FontWeight','bold');
	title('Testing Period','FontName','Arial','FontSize',12,'FontWeight','bold'); 
	xlabel('Relative Sensitivity','FontName','Arial','FontSize',12,'FontWeight','bold');
	
	xrange=[1,2,3,4,5];c = {'Fr1','Fr2','Fr3','Fr4','Fr5'};
	subplot(2,6,10); SSD = SSD_train_all(:, :, i)'./sum(SSD_train_all(:, :, i)'); bar(xrange,SSD,'Facecolor',[.5 .5 .5]);set(gca,'XTickLabel',c);
	set(gca,'linewidth',1.5,'FontName','Times New Roman','FontSize',10,'FontWeight','bold');ylim([0 0.9]);
	ylabel('% of Contribution','FontName','Arial','FontSize',10,'FontWeight','bold');%text(xrange,SSD+0.1,num2str(SSD),'Color','k');
	title('Training','FontName','Arial','FontSize',12,'FontWeight','bold'); 
	subplot(2,6,11); SSD = SSD_validation_all(:, :, i)'./sum(SSD_validation_all(:, :, i)'); bar(xrange,SSD,'Facecolor',[.5 .5 .5]);set(gca,'XTickLabel',c);
	set(gca,'linewidth',1.5,'FontName','Times New Roman','FontSize',10,'FontWeight','bold');ylim([0 0.9]);
	ylabel('% of Contribution','FontName','Arial','FontSize',10,'FontWeight','bold');%text(xrange,SSD+0.1,num2str(SSD),'Color','k');
	title('Validation','FontName','Arial','FontSize',12,'FontWeight','bold'); 
	subplot(2,6,12); SSD = SSD_test_all(:, :, i)'./sum(SSD_test_all(:, :, i)'); bar(xrange,SSD,'Facecolor',[.5 .5 .5]);set(gca,'XTickLabel',c);
	set(gca,'linewidth',1.5,'FontName','Times New Roman','FontSize',10,'FontWeight','bold');ylim([0 0.9]);
	ylabel('% of Contribution','FontName','Arial','FontSize',10,'FontWeight','bold');%text(xrange,SSD+0.1,num2str(SSD),'Color','k');
	title('Testing','FontName','Arial','FontSize',12,'FontWeight','bold'); 
end
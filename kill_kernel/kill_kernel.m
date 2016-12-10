clear
load('../Matlab_mat/VDSR_Official.mat');
load('mean_list.mat');
weight = model.weight; %1*20 cell array
bias = model.bias;
change = single(zeros(3,3,64));
for i = 1:20
    temp_weight = model.weight{1,i};
    temp_bias = model.bias{1,i};
    th1 = mean_list(i);
    th2 = 0.15;
    if i ~=20
        [height,width,channel,output] = size(temp_weight);
        temp = 0;
        sort_list=[];
        for j = 1:output
            sparsity = mean(abs(reshape(temp_weight(:,:,:,j),1,[]))<th1);
            all = all+1;
            sort_list = [sort_list [sparsity]];
        end
        sorted = sort(sort_list);
        num = int8(64*th2);
        edge = sorted(num);
        index = find(sort_list <=  edge);
        [z,length]= size(index);
        %fprintf('%d layer %d \n',i,length);
        for k = 1:length
           temp_weight(:,:,:,index(1,k)) = single(zeros(3,3,channel,1));
           temp_bias(index(1,k),1) = 0.0;
        end
        model.weight{1,i}=temp_weight;
        model.bias{1,i} = temp_bias;
    end
end
fprintf(' %.2f%% kernals were eliminated. \n',th2*100)
save('../Matlab_mat/model_15.mat','model')

%% 
for f =1:2
folder = {'S9','S11'};
% load 2d joint prediction by CPM
%load('/home/capstone/capstone126/ching/human3D/H80K/H80K/s9s11/J_p_s11_Yasin.mat');

% load NN pool
load('/home/capstone/capstone126/ching/human3D/3dpose_pool/h36m/S1_S8.mat');

% 3D ground truth
load(['/home/capstone/capstone126/ching/human3D/',folder{f},'/',folder{f},'_3D.mat']);

% 2D pixel ground truth
load(['/home/capstone/capstone126/ching/human3D/',folder{f},'/',folder{f},'_2D.mat']);

if f==1
S_2D = S9_2D;
S_3D = S9_3D;
else
    S_2D = S11_2D;
    S_3D = S11_3D;
end

count = 1;
for k = 1%[0.01 0.03 0.1 0.3 0.5 0.8]%1.0]

[trainInd,valInd,testInd] = dividerand(length(s1_s8_3d), k, 0, 1-k);
M_pool = s1_s8_2d_n(trainInd,:);
GT_pool_p = s1_s8_3d_n(trainInd,:);
GT_pool = s1_s8_3d(trainInd,:);
c1 = 1;
for i = 1:4:117%size(S_2D,1)
    
    cc = 1;
    for j = 1:size(S_2D,2)
        if ~isempty(S_2D{i,j}) && mod(j,10)==0 && j <= 1500   
        i
        j
        
        p3d = S_3D{i,j};
        p3d = reshape(p3d,3,14);
        p3d = p3d';
        
        % perspective seed
        p2d = S_2D{i,j};
        p2d = reshape(p2d,2,14);
        p2d = p2d';
        %% Given CPM2D
        %{
        %j_NN: NN of CPM2D
        %j_dual: cpmxy_nnz
        j_NN = NN_pose_baseline(i,j,M_pool,GT_pool,J_p);
        
        prediction = J_p{i,j};
        scale = (max(j_NN(:,2))-min(j_NN(:,2)))/(max(prediction(:,2))-min(prediction(:,2)));
        j_dual = ([prediction(:,1:2)*scale, j_NN(:,3)]);

        sum_dual = MPJPE(p3d,j_dual)
        cpm2d_p_n{i}(cc) = sum_dual;
        
        sum_nn_nn = MPJPE(p3d,j_NN)
        cpm2d_n_n{i}(cc) = sum_nn_nn;
        %}
        
        %% given 2D ground truth
        %%pred = s11_m_pool(cc, :);
        %p3d = NN_pose(i,j);
        
        j_2d = kNN_pose_procrus(M_pool,GT_pool,p2d,10);
        %j_np_np = kNN_pose_procrus_np(M_pool,GT_pool,p3d,10);
        %{
        pre = zeros(14,3);
        jroot=[(p3d(9,1)+p3d(12,1))/2, (p3d(9,2)+p3d(12,2))/2,(p3d(9,3)+p3d(12,3))/2];
        y_sc=zeros(14,1);
        for m = 1 : 14
            pre(m,1) = p3d(m,1) - jroot(1);
            pre(m,2) = p3d(m,2) - jroot(2);
            pre(m,3) = p3d(m,3) - jroot(3);
            y_sc(m) = pre(m, 2);
        end
        pre = pre / (max(y_sc)-min(y_sc));
        
        pred = pre(:,1:2);
        pred = reshape(pred',1,28);
        pred = double(pred);
        
        %[~,m_idx] = min(pdist2(pred, M_pool));
        m_idx = knnsearch(M_pool,pred,'k',8);
        e = zeros(1,8);
        for ii=1:8
        
        j_2d = GT_pool(m_idx(ii),:);
        j_2d = reshape(j_2d,3,14);
        j_2d = double(j_2d'); 
        
        %gt_2d = reshape(M_pool(m_idx,:),2,14);
        %gt_2d = double(gt_2d'); 
        scale = (max(j_2d(:,2))-min(j_2d(:,2)))/(max(p3d(:,2))-min(p3d(:,2)));
        j_dual_2d = ([p3d(:,1:2)*scale, j_2d(:,3)]);
        e(ii) = MPJPE_procrus(p3d(:,1:2),j_2d(:,1:2));
        
        end
        
        [~,iid] = min(e);
        j_2d = GT_pool(m_idx(iid),:);
        j_2d = reshape(j_2d,3,14);
        j_2d = double(j_2d'); 
        %}
        scale = (max(j_2d(:,2))-min(j_2d(:,2)))/(max(p3d(:,2))-min(p3d(:,2)));
        j_dual_2d = ([p3d(:,1:2)*scale, j_2d(:,3)]);
        %j_np_np = ([p3d(:,1:2)*scale, j_np_np(:,3)]);

        
        j_nn_2d = (j_2d);
        
        sum_dual_2d = MPJPE_procrus(p3d,j_dual_2d)
        gt2d_p_n{c1}(cc) = sum_dual_2d;
        %min(e)
        %gt2d_p_n{i}(cc) = min(e);
        %sum_nn_2d = MPJPE_procrus(p3d,j_nn_2d)
        %gt2d_n_n{c1}(cc) = sum_nn_2d;
        
        %% compute 3D pose error by given 3D ground truth
        %{ 
        %j_3d = s11_gt_pool(cc,:);
        %p3d = NN_pose(i,j);
        pred = pre';
        pred = reshape(pred(:),1,42);
        pred = double(pred);
        
        [~,m_idx] = min(pdist2(pred, GT_pool_p));
        j_3d = GT_pool(m_idx,:);
        j_3d = reshape(j_3d,3,14);
        j_3d = double(j_3d');
        %evaluate nn_z
        scale = (max(j_3d(:,2))-min(j_3d(:,2)))/(max(p3d(:,2))-min(p3d(:,2)));
        j_nn_z = [j_3d(:,1:2) p3d(:,3)*scale];
        j_xy_nn = [p3d(:,1:2)*scale j_3d(:,3)];
        
        sum_dual_3d = MPJPE(p3d,j_nn_z)
        sum_xy_nn = MPJPE(p3d,j_xy_nn)
        sum_n_n_3d = MPJPE(p3d,j_3d)
        
        gt3d_n_p{i}(cc) = sum_dual_3d;
        gt3d_n_n{i}(cc) = sum_n_n_3d;
        gt3d_p_n{i}(cc) = sum_xy_nn;
        %}
      
        cc = cc + 1;
        end
    end
     c1 = c1+1;
end

%st=[];for p = 1:120, st = [st,cpm2d_p_n{p}];end
%st1=[];for p = 1:120, st1 = [st1,cpm2d_n_n{p}];end

st2=[];for p = 1:30, st2 = [st2,gt2d_p_n{p}];end
st3=[];for p = 1:30, st3 = [st3,gt2d_n_n{p}];end
%st4=[];for p = 1:120, st4 = [st4,gt3d_n_p{p}];end
%st5=[];for p = 1:120, st5 = [st5,gt3d_n_n{p}];end
%st6=[];for p = 1:120, st6 = [st6,gt3d_p_n{p}];end

%CPM2D_p_n(count,2) = mean(st); CPM2D_p_n(count,1) = k; CPM2D_p_n(count,3) = median(st);
%CPM2D_n_n(count,2) = mean(st1); CPM2D_n_n(count,1) = k; CPM2D_n_n(count,3) = median(st1);

GT2D_p_n(count,2) = mean(st2); GT2D_p_n(count,1) = k; GT2D_p_n(count,3) = median(st2);
GT2D_n_n(count,2) = mean(st3); GT2D_n_n(count,1) = k; GT2D_n_n(count,3) = median(st3);
%GT3D_n_p(count,2) = mean(st4); GT3D_n_p(count,1) = k; GT3D_n_p(count,3) = median(st4);
%GT3D_n_n(count,2) = mean(st5); GT3D_n_n(count,1) = k; GT3D_n_n(count,3) = median(st5);
%GT3D_p_n(count,2) = mean(st6); GT3D_p_n(count,1) = k; GT3D_p_n(count,3) = median(st6);

count = count+1;
end
save([num2str(f),'_10nn_nnalign_3d_2d.mat'],'GT2D_p_n','GT2D_n_n','gt2d_n_n','gt2d_p_n');
clear gt2d_n_n
clear gt2d_p_n
end
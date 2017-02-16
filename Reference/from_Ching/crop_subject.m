% crop img according to centered at human subject

% load 2D position from file
%load('/home/capstone/capstone126/ching/human3D/S11/S11_2D.mat');

%file = dir(['/home/capstone/capstone126/ching/human3D/S11/s11_64th/*jpg']);
c =1;
for i = 1:size(S9_2D,1)
    for j = 1:size(S9_2D,2)
        if ~isempty(S9_2D{i,j}) && mod(j,10)==0
            im = imread(['/home/capstone/capstone126/ching/human3D/S9/s9_10fps/'...
                ,num2str(i),'_',num2str(j),'.jpg']);
            
            p2d = S9_2D{i,j};
            p2d = reshape(p2d,2,14);
            xm = max(1,uint16(min(p2d(1,:))-100)); 
            xM = min(size(im,2),uint16(max(p2d(1,:))+100));
            ym = max(1,uint16(min(p2d(2,:))-100));
            yM = min(size(im,1),uint16(max(p2d(2,:))+100));
            im = im(ym:yM,xm:xM,:);
            
            imwrite(im,['/home/capstone/capstone126/ching/human3D/S9/s9_10/',num2str(i),'_',num2str(j),'.jpg']);
        end
    end
end
%{
cc=1;
for i = 1:size(S9_2D,1)
    for j = 1:size(S9_2D,2)
        if ~isempty(S9_2D{i,j}) && mod(j,10)==0
            
            test_img = (['/home/capstone/capstone126/ching/human3D/S9/s9_10/'...
                ,num2str(i),'_',num2str(j),'.jpg']);
            
            [test_image, j_p, j_v,heatMaps] = cpmfwd(param,test_img);
            J_p_s9{i,j} = j_p;
%figure(3); imshow(test_image);
%vis_2d(j_p);
            cc=cc+1
        end
    end
    save('J_p_s9_zhou.mat','J_p_s9');
end
%}            
            
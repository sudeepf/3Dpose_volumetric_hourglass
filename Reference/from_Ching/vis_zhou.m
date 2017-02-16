% parsing output from Zhou's result
cc = 182;
Prediction = output.S_final(3*cc-2:3*cc,:);
Prediction = Prediction';

prediction(1,:)=Prediction(11,:);
prediction(2,:)=Prediction(9,:);
prediction(3,:)=Prediction(15,:);
prediction(4,:)=Prediction(16,:);
prediction(5,:)=Prediction(17,:);
prediction(6,:)=Prediction(12,:);
prediction(7,:)=Prediction(13,:);
prediction(8,:)=Prediction(14,:);
prediction(9,:)=Prediction(5,:);
prediction(10,:)=Prediction(6,:);
prediction(11,:)=Prediction(7,:);
prediction(12,:)=Prediction(2,:);
prediction(13,:)=Prediction(3,:);
prediction(14,:)=Prediction(4,:);

%% visualize with ground plane and camera
prediction = double(prediction)*50;
H=figure(12);
subplot(1,3,[1 2]);imshow(im);
%% draw ground plane
prediction(:,3) = prediction(:,3) - min(prediction(:,3)) + 1500;
an_x_m = min(prediction(:,1));
an_x_M = max(prediction(:,1));
an_y_M = max(prediction(11,2),prediction(14,2));
x = [an_x_m-100 an_x_M+100 an_x_m-100 an_x_M+100];
y = [an_y_M-10 an_y_M-10 an_y_M+40 an_y_M+40];
z = [min(prediction(:,3)-120) min(prediction(:,3)-120) max(prediction(:,3)+120) max(prediction(:,3)+120)];
A = [x(:) y(:) z(:)];
[n,v,m,aved]=q4a(A);


%vis_2d(prediction);
subplot(1,3,3);
vis_3d(prediction);
planeplot(A,n,m)
%% draw camera
scale = 50;
P = scale*[0 0 0;0.5 0.5 0.8; 0.5 -0.5 0.8; -0.5 0.5 0.8;-0.5 -0.5 0.8];
cen = mean(prediction);P1=(P+repmat([cen(1:2), 800],[5,1]));
%P1=P1';
line([P1(1,1) P1(2,1)],[P1(1,2) P1(2,2)],[P1(1,3) P1(2,3)],'color','k')
line([P1(1,1) P1(3,1)],[P1(1,2) P1(3,2)],[P1(1,3) P1(3,3)],'color','k')
line([P1(1,1) P1(4,1)],[P1(1,2) P1(4,2)],[P1(1,3) P1(4,3)],'color','k')
line([P1(1,1) P1(5,1)],[P1(1,2) P1(5,2)],[P1(1,3) P1(5,3)],'color','k')

line([P1(2,1) P1(3,1)],[P1(2,2) P1(3,2)],[P1(2,3) P1(3,3)],'color','k')
line([P1(3,1) P1(5,1)],[P1(3,2) P1(5,2)],[P1(3,3) P1(5,3)],'color','k')
line([P1(5,1) P1(4,1)],[P1(5,2) P1(4,2)],[P1(5,3) P1(4,3)],'color','k')
line([P1(4,1) P1(2,1)],[P1(4,2) P1(2,2)],[P1(4,3) P1(2,3)],'color','k')

axis off
axis equal
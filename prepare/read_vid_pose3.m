% Read imagefiles and 3D pose

subjects = ['S1', 'S5']%, 'S6', 'S7', 'S8', 'S9'];
% Dataset Paths 
for iok=1:2:length(subjects)
    subject = subjects(iok:iok+1);
    path = ['/home/capstone/datasets/Human3.6M/Subjects/',subject];
    vPath = [path,'/Videos/'];
    pPath = [path,'/Pose/D3_Positions_mono/'];
    p2Path = [path,'/Pose/D2_Positions/'];
   
    pFiles = dir([pPath,'*cdf']);
    p2Files = dir([p2Path,'*cdf']);
    
    vFolders = dir([vPath]);
    vFolders(1:2) = [];
    dirFlag = [vFolders.isdir];
    vFolders = vFolders(dirFlag);

    system(['mkdir ', path,'/mats']);
    matPath = [path,'/mats'];
    %% Looping over all the video folders
    for ii=1:length(vFolders)
        fName = vFolders(ii).name;
        fName1 = strcat(vPath,fName);
        iName = dir([fName1,'/*jpg']);
        disp([pPath,fName,'.cdf']);
        p3D = cdfread([pPath,fName,'.cdf']);
        p3D = p3D{1};
        p2D = cdfread([p2Path,fName,'.cdf']);
        p2D = p2D{1};
        imgs = [];
        poses2 = [];
        poses3 = [];
        scales = [];
        camC = [];
        matName = [matPath,'/',fName,'.mat'];
        %% Looping over all the images and pose files 
        for j = 1:length(p3D)
            imN = iName(j*2).name;
    %         im = imread([vPath,fName,'/',imN]);
            p3d = p3D(j,:);
            p2d = p2D(j,:);
            imgs = [imgs; strcat(vPath,'/',fName,'/',imN)];
            % Extracing pose from strange data type
            pose3 = extract3DJoints(p3d);
            pose2 = extract2DJoints(p2d);

            %Centering the pose2d 
            %[H W C] = size(im);
            %Computing Camera Params
            base2_1 = pose2(9,:);
            base2_2 = pose2(12,:);
            base3_1 = pose3(9,1:2)/pose3(9,3);
            base3_2 = pose3(12,1:2)/pose3(12,3);

            nume_ = base2_1 - base2_2;
            denom_ = base3_1 - base3_2;
            F = nume_./denom_; % Camera parameters 
            midP = base2_1 - (base3_1/pose3(9,3)).*F ;
            pose2_ = pose2 - repmat(midP,14,1);
            pose2_;

            % Centering 3D pose wrt z axis


            %Computing Scale of the projection
            ProjScale = pose3(:,1:2) ./ pose2_;

    %%         Visualization


    %        scatter3(pose3(:,1)./ProjScale(:,1),pose3(:,2)./ProjScale(:,2),pose3(:,3)/mean(mean(ProjScale)))
    %          imshow(im)
    %         hold on,
    %         scatter(pose2(:,1),pose2(:,2))
    %         

    %        pause
            pose3 = [pose3(:,1)./ProjScale(:,1),pose3(:,2)./ProjScale(:,2),pose3(:,3)/mean(mean(ProjScale))];
            poses2(j,:,:) = pose2;
            m3_root = 0.5*([pose3(9, 1) pose3(9, 2) pose3(9,3)]+...
                [pose3(12,1) pose3(12,2) pose3(12,3)]);

            m2_root = 0.5*([pose2(9, 1) pose2(9, 2)]+...
                [pose2(12,1) pose2(12,2)]);

            pose3(:,3) = pose3(:,3) - repmat(m3_root(1,3),14,1);
            pose3(:,1:2) = pose3(:,1:2) + repmat(midP,14,1);
            poses3(j,:,:) = pose3;
            scales(j,:,:) = ProjScale;
            camC(j,:) = midP; 
        end
        save(matName,'imgs','poses2','poses3', 'scales', 'camC')
        disp(['saved',matName])
    end
end
length(vFolders)
% prepare ground truth of testing data
%folder = {'S9'};%,'S11'};
folder = {'S1','S5','S6','S7','S8'};
count = 1;
for i = 1:length(folder)
     %cd(['/home/capstone/capstone126/ching/human3D/',folder{i},'/D2_Positions/']);
    %files = dir(['/home/capstone/capstone126/ching/human3D/',folder{i},'/D2_Positions/*cdf']);
    cd(['/home/capstone/capstone126/ching/human3D/3dpose_pool/h36m2d/',folder{i},'/']);
    files = dir(['/home/capstone/capstone126/ching/human3D/3dpose_pool/h36m2d/',folder{i},'/*cdf']);
    for k = 1:length(files)
       
        p2d = cdfread(files(k).name);
        p2d = p2d{1,1};
        cla(k)=length(p2d);
        
        for j = 1:length(p2d)
            
            S9(count, 1) = p2d(j,2*16-1);
            S9(count, 3) = p2d(j,2*14-1);
            S9(count, 5) = p2d(j,2*26-1);
            S9(count, 7) = p2d(j,2*27-1);
            S9(count, 9) = p2d(j,2*28-1);
            S9(count, 11) = p2d(j,2*18-1);
            S9(count, 13) = p2d(j,2*19-1);
            S9(count, 15) = p2d(j,2*20-1);
            S9(count, 17) = p2d(j,2*2-1);
            S9(count, 19) = p2d(j,2*3-1);
            S9(count, 21) = p2d(j,2*4-1);
            S9(count, 23) = p2d(j,2*7-1);
            S9(count, 25) = p2d(j,2*8-1);
            S9(count, 27) = p2d(j,2*9-1);
            
            
            S9(count, 2) = p2d(j,2*16);
            S9(count, 4) = p2d(j,2*14);
            S9(count, 6) = p2d(j,2*26);
            S9(count, 8) = p2d(j,2*27);
            S9(count, 10) = p2d(j,2*28);
            S9(count, 12) = p2d(j,2*18);
            S9(count, 14) = p2d(j,2*19);
            S9(count, 16) = p2d(j,2*20);
            S9(count, 18) = p2d(j,2*2);
            S9(count, 20) = p2d(j,2*3);
            S9(count, 22) = p2d(j,2*4);
            S9(count, 24) = p2d(j,2*7);
            S9(count, 26) = p2d(j,2*8);
            S9(count, 28) = p2d(j,2*9);
            
            
            m_root = 0.5*([S9(count, 17) S9(count, 23)]+...
                [S9(count,18) S9(count,24)]);
            y_sc=zeros(14,1);
            for m = 1 : 14
                s9(count, 2*m-1) = S9(count,2*m-1) - m_root(1);
                s9(count, 2*m) = S9(count,2*m) - m_root(2);
                y_sc(m) = s9(count, 2*m);
            end
            s9(count,:) = s9(count,:) / (max(y_sc)-min(y_sc));
            
        %S_2D{k,j} = S9(count,:); 
        S_2d(count,:) = S9(count,:); 
        count = count + 1
        end
        
    end
end
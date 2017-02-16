% prepare ground truth of testing data
folder = {'S9'};%,'S11'};
count = 1;
for i = 1:length(folder)
     cd(['/home/capstone/capstone126/ching/human3D/',folder{i},'/D3_Positions_mono/']);
    files = dir(['/home/capstone/capstone126/ching/human3D/',folder{i},'/D3_Positions_mono/*cdf']);
    for k = 1:length(files)
        
        p3d = cdfread(files(k).name);
        p3d = p3d{1,1};
        for j = 1:length(p3d)
            
            S9(count, 2) = p3d(j,3*16-1);
            S9(count, 5) = p3d(j,3*14-1);
            S9(count, 8) = p3d(j,3*26-1);
            S9(count, 11) = p3d(j,3*27-1);
            S9(count, 14) = p3d(j,3*28-1);
            S9(count, 17) = p3d(j,3*18-1);
            S9(count, 20) = p3d(j,3*19-1);
            S9(count, 23) = p3d(j,3*20-1);
            S9(count, 26) = p3d(j,3*2-1);
            S9(count, 29) = p3d(j,3*3-1);
            S9(count, 32) = p3d(j,3*4-1);
            S9(count, 35) = p3d(j,3*7-1);
            S9(count, 38) = p3d(j,3*8-1);
            S9(count, 41) = p3d(j,3*9-1);
            
            S9(count, 1) = p3d(j,3*16-2);
            S9(count, 4) = p3d(j,3*14-2);
            S9(count, 7) = p3d(j,3*26-2);
            S9(count, 10) = p3d(j,3*27-2);
            S9(count, 13) = p3d(j,3*28-2);
            S9(count, 16) = p3d(j,3*18-2);
            S9(count, 19) = p3d(j,3*19-2);
            S9(count, 22) = p3d(j,3*20-2);
            S9(count, 25) = p3d(j,3*2-2);
            S9(count, 28) = p3d(j,3*3-2);
            S9(count, 31) = p3d(j,3*4-2);
            S9(count, 34) = p3d(j,3*7-2);
            S9(count, 37) = p3d(j,3*8-2);
            S9(count, 40) = p3d(j,3*9-2);
            
            S9(count, 3) = p3d(j,3*16);
            S9(count, 6) = p3d(j,3*14);
            S9(count, 9) = p3d(j,3*26);
            S9(count, 12) = p3d(j,3*27);
            S9(count, 15) = p3d(j,3*28);
            S9(count, 18) = p3d(j,3*18);
            S9(count, 21) = p3d(j,3*19);
            S9(count, 24) = p3d(j,3*20);
            S9(count, 27) = p3d(j,3*2);
            S9(count, 30) = p3d(j,3*3);
            S9(count, 33) = p3d(j,3*4);
            S9(count, 36) = p3d(j,3*7);
            S9(count, 39) = p3d(j,3*8);
            S9(count, 42) = p3d(j,3*9);
            
            
            m_root = 0.5*([S9(count, 25) S9(count, 26)]+...
                [S9(count,34) S9(count,35)]);
            y_sc=zeros(14,1);
            for m = 1 : 14
                s9(count, 2*m-1) = S9(count,3*m-2) - m_root(1);
                s9(count, 2*m) = S9(count,3*m-1) - m_root(2);
                y_sc(m) = s9(count, 2*m);
            end
            s9(count,:) = s9(count,:) / (max(y_sc)-min(y_sc));
            S9_3D{k,j} = S9(count,:);
        
        count = count + 1
        end
    end
end
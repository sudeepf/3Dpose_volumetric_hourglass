%% Readin Videos and Extract frames to a folder

subject = 'S1';

path = ['../Dataset/',subject];
vPath = [path,'/Videos/'];
pPath = [path,'/Pose/D3_Positions_mono/'];

pFiles = dir([pPath,'*cdf']);
vFiles = dir([vPath, '*mp4']);

for ii = 1:length(vFiles)
    fname = vFiles(ii).name;
    if(fname(1) == '_')
        continue,
    end
    ll = strsplit(fname,'.m')
    ll = ll(1)
    disp(strjoin(['mkdir ',vPath,ll],''))
    system(strjoin(['mkdir ',vPath,ll],''))
    ll = cell2mat(ll)
    lol = [vPath,ll,'/frame%4d.jpg']
    disp(['ffmpeg  -i ', [vPath,fname],' -r 100.0 ',  lol])
    system(['ffmpeg  -i ', [vPath,fname],' -r 100.0 ',  lol])
end

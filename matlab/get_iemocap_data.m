% Read the .arff files in the arffdir according to the label file
% Output the emotion vectors in rows (x) and classID (y) with ID starts from 0
% Also output the targets in one-of-K format
% Example:
%   labels = {'ang','hap','neu','sad'};
%   classes = {1,2,3,4};
%   labmap = containers.Map(labels,classes);
%   dat = get_iemocap_data('../arff/IS09_emotion', '../labels/emo_labels_cat.txt', labmap);
function dat = get_iemocap_data(arffdir, labfile, labmap)

nlines = numlines(labfile);
%nlines = 10;

% Get the file info of all files under the arff folder
x = []; y = []; targets = []; spknums = []; genders = [];

nclasses = length(keys(labmap));
fid = fopen(labfile,'rt');
j = 1;
for i=1:nlines,
    line = fgetl(fid);
    field = rsplit('\s+', line);
    if isKey(labmap,field{2}),
        arfffile = sprintf('%s/%s.arff', arffdir,field{1});
        fprintf('Reading %s with label %s\n',arfffile,field{2});
        wekaOBJ = loadARFF(arfffile);
        mdata = weka2matlab(wekaOBJ);
        data = mdata(:,2:end-1);      % data is an 1 x nFeatures matrix
        x = [x; data];
        lab{1} = field{2};
        id = values(labmap,lab);
        y = [y; id{1}-1];
        tgt = zeros(1,nclasses);
        tgt(id{1}) = 1;
        targets = [targets; tgt];
        spknums = [spknums; str2num(field{3})];
        genders = [genders; lower(field{4})];
        j = j + 1;
    end
end

% return structure
dat.x = x;
dat.y = y;
dat.targets = targets;
dat.nclasses = nclasses;
dat.spknums = spknums;
dat.genders = genders;







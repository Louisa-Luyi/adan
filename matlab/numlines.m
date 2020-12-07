function nlines = numlines(filename)
% Return the no. of lines in the input file

if (exist(filename,'file'))==0,
    error('File %s not found',filename);
end
fid = fopen(filename,'rt');
nlines = 0;
while (fgets(fid) ~= -1),
  nlines = nlines+1;
end
fclose(fid);

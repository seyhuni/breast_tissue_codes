% split classes
class1 = cancerTargetsSingleRow(1, 1:21);
class2 = cancerTargetsSingleRow(1, 22:36);
class3 = cancerTargetsSingleRow(1, 37:54);
class4 = cancerTargetsSingleRow(1, 55:70);
class5 = cancerTargetsSingleRow(1, 71:84);
class6 = cancerTargetsSingleRow(1, 85:106);

data1 = cancerInputs(:, 1:21);
data2 = cancerInputs(:, 22:36);
data3 = cancerInputs(:, 37:54);
data4 = cancerInputs(:, 55:70);
data5 = cancerInputs(:, 71:84);
data6 = cancerInputs(:, 85:106);

[c1X, c1Y] = size(class1);
[c2X, c2Y] = size(class2);
[c3X, c3Y] = size(class3);
[c4X, c4Y] = size(class4);
[c5X, c5Y] = size(class5);
[c6X, c6Y] = size(class6);

% train data count
c1rat = round(ratio * c1Y);
c2rat = round(ratio * c2Y);
c3rat = round(ratio * c3Y);
c4rat = round(ratio * c4Y);
c5rat = round(ratio * c5Y);
c6rat = round(ratio * c6Y);

% test data count
tc1rat = c1Y - c1rat;
tc2rat = c2Y - c2rat;
tc3rat = c3Y - c3rat;
tc4rat = c4Y - c4rat;
tc5rat = c5Y - c5rat;
tc6rat = c6Y - c6rat;

t1inds = randperm(c1Y, tc1rat);
t2inds = randperm(c2Y, tc2rat);
t3inds = randperm(c3Y, tc3rat);
t4inds = randperm(c4Y, tc4rat);
t5inds = randperm(c5Y, tc5rat);
t6inds = randperm(c6Y, tc6rat);

valid1TestD = data1(:, t1inds);
valid2TestD = data2(:, t2inds);
valid3TestD = data3(:, t3inds);
valid4TestD = data4(:, t4inds);
valid5TestD = data5(:, t5inds);
valid6TestD = data6(:, t6inds);

data1(:, t1inds) = [];
data2(:, t2inds) = [];
data3(:, t3inds) = [];
data4(:, t4inds) = [];
data5(:, t5inds) = [];
data6(:, t6inds) = [];

target1TestL = class1(t1inds);
target2TestL = class2(t2inds);
target3TestL = class3(t3inds);
target4TestL = class4(t4inds);
target5TestL = class5(t5inds);
target6TestL = class6(t6inds);

class1(t1inds) = [];
class2(t2inds) = [];
class3(t3inds) = [];
class4(t4inds) = [];
class5(t5inds) = [];
class6(t6inds) = [];

train1D = [data1 data2 data3 data4 data5 data6];
train1L = categorical([class1 class2 class3 class4 class5 class6]');

trainTest = [valid1TestD valid2TestD valid3TestD valid4TestD valid5TestD valid6TestD];
targetTestL = categorical([target1TestL target2TestL target3TestL target4TestL target5TestL target6TestL]');

clearvars -except cancerInputs cancertargets cancerTargetsSingleRow catCancerClasses ratio targetTestL train1D train1L trainTest
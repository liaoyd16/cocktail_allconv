colors = zeros(6,3);
colors(1,:) = [1,0,0];
colors(2,:) = [0,0.7,0];
colors(3,:) = [0.5,0.5,0];
colors(4,:) = [0,0,1];
colors(5,:) = [1,0,1];
colors(6,:) = [0,.7,.7];

names = ["1-0","1-1", "2-0","2-2", "3-0","3-3"];

% to delete unfit curves
delete(hL); delete(eBar);


%% n5cumu
x1 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=5\en=1_de=0.mat').ami_array;
x2 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=5\en=1_de=en.mat').ami_array;
x3 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=5\en=2_de=0.mat').ami_array;
x4 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=5\en=2_de=en.mat').ami_array;
x5 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=5\en=3_de=0.mat').ami_array;
x6 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=5\en=3_de=en.mat').ami_array;

% fitting
[fitresult1, gof1,  hL, eBar] = sigmoid_fit(x1', true, colors(1,:), [5,rand(),rand()]);
[fitresult2, gof2,  hL, eBar] = sigmoid_fit(x2', true, colors(2,:), [4,rand(),rand()]);
[fitresult3, gof3,  hL, eBar] = sigmoid_fit(x3', true, colors(3,:), [4,rand(),rand()]);
[fitresult4, gof4,  hL, eBar] = sigmoid_fit(x4', true, colors(4,:), [5,rand(),rand()]);
[fitresult5, gof5,  hL, eBar] = sigmoid_fit(x5', true, colors(5,:), [3.7,rand(),rand()]);
[fitresult6, gof6,  hL, eBar] = sigmoid_fit(x6', true, colors(6,:), [3.5,rand(),rand()]);

% save fitting
n5cumu = {fitresult1, fitresult2, fitresult3, fitresult4, fitresult5, fitresult6, gof1, gof2, gof3, gof4, gof5, gof6};


%% n7cumu
x1 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=7\en=1_de=0.mat').ami_array;
x2 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=7\en=1_de=en.mat').ami_array;
x3 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=7\en=2_de=0.mat').ami_array;
x4 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=7\en=2_de=en.mat').ami_array;
x5 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=7\en=3_de=0.mat').ami_array;
x6 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=7\en=3_de=en.mat').ami_array;

% fitting
[fitresult1, gof1,  hL, eBar] = sigmoid_fit(x1', true, colors(1,:), [4.5,rand(),rand()]);
[fitresult2, gof2,  hL, eBar] = sigmoid_fit(x2', true, colors(2,:), [4.5,rand(),rand()]);
[fitresult3, gof3,  hL, eBar] = sigmoid_fit(x3', true, colors(3,:), [5,rand(),rand()]);
[fitresult4, gof4,  hL, eBar] = sigmoid_fit(x4', true, colors(4,:), [5,rand(),rand()]);
[fitresult5, gof5,  hL, eBar] = sigmoid_fit(x5', true, colors(5,:), [5,rand(),rand()]);
[fitresult6, gof6,  hL, eBar] = sigmoid_fit(x6', true, colors(6,:), [5,rand(),rand()]);

% save fitting
n7cumu = {fitresult1, fitresult2, fitresult3, fitresult4, fitresult5, fitresult6, gof1, gof2, gof3, gof4, gof5, gof6};


%% n10cumu
x1 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=10\en=1_de=0.mat').ami_array;
x2 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=10\en=1_de=en.mat').ami_array;
x3 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=10\en=2_de=0.mat').ami_array;
x4 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=10\en=2_de=en.mat').ami_array;
x5 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=10\en=3_de=0.mat').ami_array;
x6 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_cumu-tpdwn\n=10\en=3_de=en.mat').ami_array;

% fitting
[fitresult1, gof1,  hL, eBar] = sigmoid_fit(x1', true, colors(1,:), [7.5,rand(),rand()]);
[fitresult2, gof2,  hL, eBar] = sigmoid_fit(x2', true, colors(2,:), [8,rand(),rand()]);
[fitresult3, gof3,  hL, eBar] = sigmoid_fit(x3', true, colors(3,:), [8,rand(),rand()]);
[fitresult4, gof4,  hL, eBar] = sigmoid_fit(x4', true, colors(4,:), [8,rand(),rand()]);
[fitresult5, gof5,  hL, eBar] = sigmoid_fit(x5', true, colors(5,:), [9,rand(),rand()]);
[fitresult6, gof6,  hL, eBar] = sigmoid_fit(x6', true, colors(6,:), [10,rand(),rand()]);

% save fitting
n10cumu = {fitresult1, fitresult2, fitresult3, fitresult4, fitresult5, fitresult6, gof1, gof2, gof3, gof4, gof5, gof6};


%% n5single
x1 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=5\en=1_de=0.mat').ami_array;
x2 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=5\en=1_de=en.mat').ami_array;
x3 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=5\en=2_de=0.mat').ami_array;
x4 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=5\en=2_de=en.mat').ami_array;
x5 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=5\en=3_de=0.mat').ami_array;
x6 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=5\en=3_de=en.mat').ami_array;

% fitting
[fitresult1, gof1,  hL, eBar] = sigmoid_fit(x1', true, colors(1,:), [3.5,rand(),rand()]);
[fitresult2, gof2,  hL, eBar] = sigmoid_fit(x2', true, colors(2,:), [4,rand(),rand()]);
[fitresult3, gof3,  hL, eBar] = sigmoid_fit(x3', true, colors(3,:), [4,rand(),rand()]);
[fitresult4, gof4,  hL, eBar] = sigmoid_fit(x4', true, colors(4,:), [4,rand(),rand()]);
[fitresult5, gof5,  hL, eBar] = sigmoid_fit(x5', true, colors(5,:), [4,rand(),rand()]);
[fitresult6, gof6,  hL, eBar] = sigmoid_fit(x6', true, colors(6,:), [4,rand(),rand()]);

% save fitting
n5single = {fitresult1, fitresult2, fitresult3, fitresult4, fitresult5, fitresult6, gof1, gof2, gof3, gof4, gof5, gof6};


%% n7single
x1 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=7\en=1_de=0.mat').ami_array;
x2 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=7\en=1_de=en.mat').ami_array;
x3 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=7\en=2_de=0.mat').ami_array;
x4 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=7\en=2_de=en.mat').ami_array;
x5 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=7\en=3_de=0.mat').ami_array;
x6 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=7\en=3_de=en.mat').ami_array;

% fitting
[fitresult1, gof1,  hL, eBar] = sigmoid_fit(x1', true, colors(1,:), [4,rand(),rand()]);
[fitresult2, gof2,  hL, eBar] = sigmoid_fit(x2', true, colors(2,:), [4,rand(),rand()]);
[fitresult3, gof3,  hL, eBar] = sigmoid_fit(x3', true, colors(3,:), [4,rand(),rand()]);
[fitresult4, gof4,  hL, eBar] = sigmoid_fit(x4', true, colors(4,:), [4,rand(),rand()]);
[fitresult5, gof5,  hL, eBar] = sigmoid_fit(x5', true, colors(5,:), [4,rand(),rand()]);
[fitresult6, gof6,  hL, eBar] = sigmoid_fit(x6', true, colors(6,:), [4,rand(),rand()]);

% save fitting
n7single = {fitresult1, fitresult2, fitresult3, fitresult4, fitresult5, fitresult6, gof1, gof2, gof3, gof4, gof5, gof6};


%% n10single
x1 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=10\en=1_de=0.mat').ami_array;
x2 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=10\en=1_de=en.mat').ami_array;
x3 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=10\en=2_de=0.mat').ami_array;
x4 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=10\en=2_de=en.mat').ami_array;
x5 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=10\en=3_de=0.mat').ami_array;
x6 = load('C:\Users\Lenovo\Desktop\cocktail_windows\mat\mat\condition_single-tpdwn\n=10\en=3_de=en.mat').ami_array;

% fitting
[fitresult1, gof1,  hL, eBar] = sigmoid_fit(x1', true, colors(1,:), [9,rand(),rand()]);
[fitresult2, gof2,  hL, eBar] = sigmoid_fit(x2', true, colors(2,:), [9,rand(),rand()]);
[fitresult3, gof3,  hL, eBar] = sigmoid_fit(x3', true, colors(3,:), [10,rand(),rand()]);
[fitresult4, gof4,  hL, eBar] = sigmoid_fit(x4', true, colors(4,:), [9,rand(),rand()]);
[fitresult5, gof5,  hL, eBar] = sigmoid_fit(x5', true, colors(5,:), [10,rand(),rand()]);
[fitresult6, gof6,  hL, eBar] = sigmoid_fit(x6', true, colors(6,:), [11,5*rand(),rand()]);

% save fitting
n10single = {fitresult1, fitresult2, fitresult3, fitresult4, fitresult5, fitresult6, gof1, gof2, gof3, gof4, gof5, gof6};
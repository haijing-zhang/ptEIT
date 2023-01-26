clc;
exp_pre = permute(reshape(readmatrix('exp_pre.txt'), [3, 500, 3]),[1,3,2]);
expinform = readtable(['exp_info.csv']);
exp_cdt = table2array(expinform(1:3, ('pre')));
exp_num = table2array(expinform(1, ('num')));
exp_fre = table2array(expinform(1, ('fre')));
g = [0, 0.75, 0.75];
figure1 = figure('WindowState','maximized');
markersize = 40;
if exp_cdt(1) == 1
    scatter3(exp_pre(1,1,:),exp_pre(1,2,:),exp_pre(1,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on; 
elseif exp_cdt(1) == 2
    scatter3(exp_pre(1,1,:),exp_pre(1,2,:),exp_pre(1,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end
if exp_cdt(2) == 1
    scatter3(exp_pre(2,1,:),exp_pre(2,2,:),exp_pre(2,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on;
elseif exp_cdt(2) == 2
    scatter3(exp_pre(2,1,:),exp_pre(2,2,:),exp_pre(2,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end
% exp_cdt(3) = 0;
if exp_cdt(3) == 1
    scatter3(exp_pre(3,1,:),exp_pre(3,2,:),exp_pre(3,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on;
elseif exp_cdt(3) == 2
    scatter3(exp_pre(3,1,:),exp_pre(3,2,:),exp_pre(3,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end
title(sprintf('Experiment %dth %dk Reconstruction', exp_num, exp_fre));
m = 200;
R = 15;
[x,y,z]=cylinder(R, m);
z=20*z;
p=surf(x,y,z);
set(p,'FaceColor','b','EdgeColor','none');
set(p,'FaceAlpha',0.1);
axis equal 
axis off
x0 = 250;
y0 = 150;
width=1000;
height=500;
set(gcf,'position',[x0, y0, width, height]);
%%
v_exp_origin = load('v_exp_origin.txt');
v_exp = load('v_exp.txt');
figure(2)
subplot(2,1,1)
plot(v_exp_origin, 'm*--','LineWidth',2);
legend('Orginal normalized data');
axis tight

subplot(2,1,2)
plot(v_exp,'m*--','LineWidth',2);
legend('Extracted nomalized data');
axis tight


clc;
sim_gt = permute(reshape(readmatrix('sim_gt.txt'), [3, 500, 3]),[1,3,2]);
sim_pre = permute(reshape(readmatrix('sim_pre.txt'), [3, 500, 3]),[1,3,2]);
inform = readtable('information.csv');
num = table2array(inform(1,('num')));
SNR = table2array(inform(1,('SNR')));
CD = table2array(inform(1,('CD')));
HD = table2array(inform(1,('HD')));
Crossentropy = table2array(inform(1,('crossentropy')));
accuracy = table2array(inform(1, ('accuracy')));
cdt_gt = table2array(inform(1:3, ('gt')));
cdt_pre = table2array(inform(1:3, ('pre')));

g = [0, 0.75, 0.75];
figure1 = figure('WindowState','maximized');

subplot1 = subplot(1,2,1,'Parent',figure1);
markersize = 40;
if cdt_gt(1) == 1
    scatter3(sim_gt(1,1,:),sim_gt(1,2,:),sim_gt(1,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on; 
elseif cdt_gt(1) == 2
    scatter3(sim_gt(1,1,:),sim_gt(1,2,:),sim_gt(1,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end

if cdt_gt(2) == 1
    scatter3(sim_gt(2,1,:),sim_gt(2,2,:),sim_gt(2,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on;
elseif cdt_gt(2) == 2
    scatter3(sim_gt(2,1,:),sim_gt(2,2,:),sim_gt(2,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end

if cdt_gt(3) == 1
    scatter3(sim_gt(3,1,:),sim_gt(3,2,:),sim_gt(3,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on;
elseif cdt_gt(3) == 2
    scatter3(sim_gt(3,1,:),sim_gt(3,2,:),sim_gt(3,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end
m = 200;
R = 15;
[x,y,z]=cylinder(R, m);
z=20*z;
p=surf(x,y,z);
set(p,'FaceColor','b','EdgeColor','none');
set(p,'FaceAlpha',0.1);
axis equal 
axis off
subtitle(sprintf('Test %dth Ground Truth', num));


subplot2 = subplot(1,2,2,'Parent',figure1);
if cdt_pre(1) == 1
    scatter3(sim_pre(1,1,:),sim_pre(1,2,:),sim_pre(1,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on; 
elseif cdt_gt(1) == 2
    scatter3(sim_pre(1,1,:),sim_pre(1,2,:),sim_pre(1,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end

if cdt_pre(2) == 1
    scatter3(sim_pre(2,1,:),sim_pre(2,2,:),sim_pre(2,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on;
elseif cdt_gt(2) == 2
    scatter3(sim_pre(2,1,:),sim_pre(2,2,:),sim_pre(2,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end

if cdt_pre(3) == 1
    scatter3(sim_pre(3,1,:),sim_pre(3,2,:),sim_pre(3,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
    hold on;
elseif cdt_pre(3) == 2
    scatter3(sim_pre(3,1,:),sim_pre(3,2,:),sim_pre(3,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor', g);
    hold on;
end

m = 200;
R = 15;
[x,y,z]=cylinder(R, m);
z=20*z;
p=surf(x,y,z);
set(p,'FaceColor','b','EdgeColor','none');
set(p,'FaceAlpha',0.1);
axis equal 
axis off
subtitle(sprintf('Test %dth Reconstruction', num));
x0 = 250;
y0 = 150;
width=1000;
height=500;
set(gcf,'position',[x0, y0, width, height]);
text('Parent',subplot2,'FontSize',12, 'String',['Signal Noise Ratio: ' num2str(SNR)],...
    'Position',[-61.6421277308983 68.2877332003635 27.6260269325981],...
    'Visible','on');
text('Parent',subplot2,'FontSize',12,'String',['Chamfer distance: ' num2str(CD)],...
    'Position',[-61.6421277308983 68.2877332003635 24.6260269325981],...
    'Visible','on');
text('Parent',subplot2,'FontSize',12,'String',['Hausdorff distance: ' num2str(HD)],...
    'Position',[-61.6421277308983 68.2877332003635 21.6260269325981],...
    'Visible','on');
% text('Parent',subplot2,'String',['Crossentropy: ' num2str(Crossentropy)],...
%     'Position',[-61.6421277308983 68.2877332003635 21.6260269325981],...
%     'Visible','on');
text('Parent',subplot2,'FontSize',12,'String',['Conductivity Accuracy: ' num2str(accuracy*100) '%'],...
    'Position',[-61.6421277308983 68.2877332003635 18.6260269325981],...
    'Visible','on');
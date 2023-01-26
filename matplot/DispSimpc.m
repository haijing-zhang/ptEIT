function DispSimpc(sim_gt, sim_pre)
figure()
subplot(1,2,1);
markersize = 40;
scatter3(sim_gt(1,1,:),sim_gt(1,2,:),sim_gt(1,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
hold on; 
scatter3(sim_gt(2,1,:),sim_gt(2,2,:),sim_gt(2,3,:), markersize, 'MarkerEdgeColor','b', 'MarkerFaceColor','w');
hold on;

m = 200;
R = 15;
[x,y,z]=cylinder(R, m);
z=20*z;
p=surf(x,y,z);
set(p,'FaceColor','b','EdgeColor','none');
set(p,'FaceAlpha',0.1);
axis equal 
axis off
subtitle('simulation sample ground truth')

subplot(1,2,2);
scatter3(sim_pre(1,1,:),sim_pre(1,2,:),sim_pre(1,3,:), markersize, 'MarkerEdgeColor','g', 'MarkerFaceColor','w');
hold on; 
scatter3(sim_pre(2,1,:),sim_pre(2,2,:),sim_pre(2,3,:), markersize, 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
hold on;
m = 200;
R = 15;
[x,y,z]=cylinder(R, m);
z=20*z;
p=surf(x,y,z);
set(p,'FaceColor','b','EdgeColor','none');
set(p,'FaceAlpha',0.1);
axis equal 
axis off
subtitle('simulation sample ground truth')
x0 = 250;
y0 = 150;
width=1000;
height=500;
set(gcf,'position',[x0, y0, width, height]);
end


fig = openfig("figures/n=5_cumu.fig");
h = get(gca, 'Children');
set(h(1),'YNegativeDelta',[],'YPositiveDelta',[]);
set(h(3),'YNegativeDelta',[],'YPositiveDelta',[]);
set(h(5),'YNegativeDelta',[],'YPositiveDelta',[]);
set(h(7),'YNegativeDelta',[],'YPositiveDelta',[]);
set(h(9),'YNegativeDelta',[],'YPositiveDelta',[]);
set(h(11),'YNegativeDelta',[],'YPositiveDelta',[]);
ylim([0,1.5]);
xticks([]);
yticks([]);
xlabel("");
ylabel("");

set(h(1), 'Color', [0.5,0,0]);
set(h(2), 'Color', [0.5,0,0]);
set(h(2), 'linewidth', 1);

set(h(3), 'Color', [0.75,0,0]);
set(h(4), 'Color', [0.75,0,0]);
set(h(4), 'linewidth', 1);

set(h(5), 'Color', [1,0,1]);
set(h(6), 'Color', [1,0,1]);
set(h(6), 'linewidth', 1);

set(h(7), 'Color', [0,0,.75]);
set(h(8), 'Color', [0,0,.75]);
set(h(8), 'linewidth', 1);

set(h(9), 'Color', [0,.5,.5]);
set(h(10), 'Color', [0,.5,.5]);
set(h(10), 'linewidth', 1);

set(h(11), 'Color', [0,0,.25]);
set(h(12), 'Color', [0,0,.25]);
set(h(12), 'linewidth', 1);

set(h(1), 'marker', 'x')
set(h(3), 'marker', '>')
set(h(5), 'marker', 'square')
set(h(7), 'marker', 'diamond')
set(h(9), 'marker', '+')
set(h(11), 'marker', 'o')

set(gca,'Fontsize',12);
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 3.5, 3.5], 'PaperUnits', 'Inches', 'PaperSize', [3.5, 3.5]);
lgd = legend([h(2),h(4),h(6),h(8),h(10),h(12)], {'k=3,m=3','k=3,m=0','k=2,m=2','k=2,m=0','k=1,m=1','k=1,m=0'});
set(lgd,'Fontname', 'Arial','FontWeight','normal','FontSize',14);

print('-dtiff','-r500',"figures/n=5_cumu.png");
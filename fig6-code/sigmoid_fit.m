function [fitresult, gof, hL,eBar] = sigmoid_fit(ami_groups_raw, do_sigmoid, color, startp, model_name) % size = (n_samples, n_attentions)
    n_grps = size(ami_groups_raw, 2);
    att_groups = [];
    ami_groups = [];
    for igrp=1:n_grps
        att_groups = [att_groups, (igrp-1) * ones(1, size(ami_groups_raw(:,igrp),1))];
        ami_groups = [ami_groups, reshape(ami_groups_raw(:,igrp), [1, size(ami_groups_raw(:,igrp),1)])];
    end
    
    [xData, yData] = prepareCurveData(att_groups, ami_groups);
    if (do_sigmoid)
        ft = fittype( 'c/(1+exp((x-a)*b))', 'independent', 'x', 'dependent', 'y' );
    else
        ft = fittype( 'e*x+f', 'independent', 'x', 'dependent', 'y' );
    end
    opts = fitoptions( 'Method', 'NonlinearLeastSquares', 'StartPoint', startp);
    [fitresult, gof] = fit( xData, yData, ft, opts );

    hL = plot(fitresult); set(hL, 'Color', color); hold on; 
    eBar = errorbar(0:n_grps-1, mean(ami_groups_raw), std(ami_groups_raw), 'x', 'Color', color); hold on;
    % lgd = legend({model_name},'Location','northwest'); hold on;
end
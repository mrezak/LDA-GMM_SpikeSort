function gscatter3(X,g,colors,fighand)
    if nargin<2 || isempty(g)
        g=ones(1,size(X,2));
    end

    g=g(:)';
    cL = unique(g);
    if ~exist('colors','var') || isempty(colors)
        colors = lines(length(cL));
    end
    colors = [colors(end,:);colors(1:end-1,:)];

    i = 0;
    if ~exist('fighand','var')
        figure
        hold on
    else
        axis(fighand);
        hold on
    end
    
    for c = cL
        i=i+1;
            if size(X,1)>2
                j = g==c;
                plot3(X(1,j),X(2,j),X(3,j),'.','color',colors(i,:),'LineWidth',2);
            elseif size(X,1)==2
                j = g==c;
                plot(X(1,j),X(2,j),'.','color',colors(i,:),'LineWidth',2);
            elseif size(X,1)==1
                j = g==c;
                B = linspace(min(X),max(X),100);
                [V,H] = hist(X(1,j),B);
                hnd = bar(H,V);
                set(hnd,'FaceColor',colors(i,:));
            end

    end
    hold off
    if size(X,1) > 1
            xlim(mean(X(1,:))+[-3*std(X(1,:)),3*std(X(1,:))])
            ylim(mean(X(2,:))+[-3*std(X(2,:)),3*std(X(2,:))])
    end
end
library(RColorBrewer)  
library(ggpubr)
library(penalized)
library(glmnet)
library(lattice)
library(ggsci)
library(tidyr)
library(ggplot2)
library(extrafont)

score_davies = read.csv("./Results/score_optimal_nb_clusters.csv")

to_plot = data.frame("Number_of_clusters"=c(2:15), "Davies_Bouldin_index"=score_davies$davies_bouldin_score)
ggline(to_plot, x="Number_of_clusters",y="Davies_Bouldin_index", color="steelblue",size=1, ggtheme=theme_classic(),xlab="Number of clusters", ylab="Davies-Bouldin index")+ theme(text=element_text(size=7, family="sans")) + scale_x_continuous(breaks = to_plot$Number_of_clusters) + geom_segment(aes(x=8, y=2.9, xend=8, yend=Davies_Bouldin_index[7]), linetype="dashed", size=1)
ggline(to_plot, x="Number_of_clusters",y="Davies_Bouldin_index", color="steelblue",size=1, ggtheme=theme_classic()) +
  scale_x_continuous(breaks = round(seq(min(to_plot$Number_of_clusters), max(to_plot$Number_of_clusters), by = 1), 1))

data_box_plots = read.csv("./Results/percentage_clusters.csv")

my_pairs = list(c(c('percentage_cluster0', 'Positive Response'),
             c('percentage_cluster0', 'Negative Response')),
            c(c('percentage_cluster1', 'Positive Response'),
             c('percentage_cluster1', 'Negative Response')),
            c(c('percentage_cluster2', 'Positive Response'),
             c('percentage_cluster2', 'Negative Response')),
            c(c('percentage_cluster3', 'Positive Response'),
             c('percentage_cluster3', 'Negative Response')),
            c(c('percentage_cluster4', 'Positive Response'),
             c('percentage_cluster4', 'Negative Response')),
            c(c('percentage_cluster5', 'Positive Response'),
             c('percentage_cluster5', 'Negative Response')),
            c(c('percentage_cluster6', 'Positive Response'),
             c('percentage_cluster6', 'Negative Response')),
            c(c('percentage_cluster7', 'Positive Response'),
             c('percentage_cluster7', 'Negative Response')))


box_plot <-ggboxplot(data_box_plots, x = "Feature.names", y = "value",
             color = "Objective.Response", palette = "npg",
             add = "point", xlab="Feature names", ylab="Value") + labs(color='Objective response to treatment :') 
box_plot + stat_compare_means(aes(group=Objective.Response), label = "p.format") + scale_x_discrete(labels = c("Percentage cluster 0", "Percentage cluster 1", "Percentage cluster 2", "Percentage cluster 3", "Percentage cluster 4", "Percentage cluster 5", "Percentage cluster 6", "Percentage cluster 7")) 

coul = c("#E41A1C", "#377EB8",  "#4DAF4A", "#984EA3", "#FFFF33", "#A65628", "#F781BF", "#999999")
names = c("Percentage cluster 0", "Percentage cluster 1", "Percentage cluster 2", "Percentage cluster 3", "Percentage cluster 4", "Percentage cluster 5", "Percentage cluster 6", "Percentage cluster 7")
val = c(0.16164396128617953,
        0.03720012843447548,
        0.17297371680198156,
        0.04889683959451401,
        0.31466446493280126,
        0.06219898169808724,
        0.19347736342369615,
        0.008944543828264758)
barplot(height=val, names=names, col=coul, horiz=T, las=1)

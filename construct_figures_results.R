library(ggpubr)
library(plyr)
library(rstatix)

# COHORT 1 analysis

## Optimal number of clusters

score_davies = read.csv("./Results/clustering/score_optimal_nb_clusters_davies_bouldin_cohort1.csv")

to_plot_davies = data.frame("Number_of_clusters"=c(7:12), "Davies_Bouldin_index"=score_davies$davies_bouldin_score)

ggline(to_plot_davies, x="Number_of_clusters",y="Davies_Bouldin_index", color="steelblue",size=1, ggtheme=theme_classic(),
       xlab="Number of clusters", ylab="Davies-Bouldin index", ylim=c(3.6,4.2)) +
  theme(axis.text=element_text(size=10), axis.title=element_text(size=12)) +
  scale_x_continuous(breaks = to_plot_davies$Number_of_clusters) +
  geom_segment(aes(x=8, y=2.8, xend=8, yend=Davies_Bouldin_index[2]), linetype="dashed", size=1)
#ggline(to_plot, x="Number_of_clusters",y="Davies_Bouldin_index", color="steelblue",size=1, ggtheme=theme_classic()) +
#  scale_x_continuous(breaks = round(seq(min(to_plot$Number_of_clusters), max(to_plot$Number_of_clusters), by = 1), 1))

## Box plots of clusters' percentage

my_pairs = list(c(c('percentage_cluster0', 'Response'),
                  c('percentage_cluster0', 'Non-response')),
                c(c('percentage_cluster1', 'Response'),
                  c('percentage_cluster1', 'Non-response')),
                c(c('percentage_cluster2', 'Response'),
                  c('percentage_cluster2', 'Non-response')),
                c(c('percentage_cluster3', 'Response'),
                  c('percentage_cluster3', 'Non-response')),
                c(c('percentage_cluster4', 'Response'),
                  c('percentage_cluster4', 'Non-response')),
                c(c('percentage_cluster5', 'Response'),
                  c('percentage_cluster5', 'Non-response')),
                c(c('percentage_cluster6', 'Response'),
                  c('percentage_cluster6', 'Non-response')),
                c(c('percentage_cluster7', 'Response'),
                  c('percentage_cluster7', 'Non-response')))

data_box_plots = read.csv("./Results/clustering/percentage_clusters_cohort_1.csv")
data_box_plots$Objective.Response <- mapvalues(data_box_plots$Objective.Response,
                                               from=c("Negative Response","Positive Response"),
                                               to=c("Non-response","Response"))

box_plot <-ggboxplot(data_box_plots, x = "Feature.names", y = "value",
                     color = "Objective.Response", palette = "npg",
                     add = "point", xlab="Cluster id", ylab="Cluster percentage") + labs(color='Objective response to treatment :') 

stat.test <- data_box_plots %>%
  group_by(Feature.names) %>%
  wilcox_test(value ~ Objective.Response)
stat.test$p.adj = format(round(p.adjust(stat.test$p, method = "BH", n = length(stat.test$p)), 4), nsmall=4)
stat.test <- stat.test %>% 
  add_xy_position(x = "Feature.names", dodge = 0.8)

box_plot + stat_pvalue_manual(stat.test, label="p = {p.adj}", tip.length = 0.01) +
  scale_x_discrete(labels = c("0", "1", "2", "3", "4", "5", "6", "7"))  + 
  scale_y_continuous(labels = scales::percent, breaks=c(0,.2,.4,.6,.8), limits=c(0,.8)) +
  theme(axis.text=element_text(size=10), axis.title=element_text(size=12))
#box_plot + stat_compare_means(aes(group=Objective.Response), label = "p.format") + scale_x_discrete(labels = c("Percentage cluster 0", "Percentage cluster 1", "Percentage cluster 2", "Percentage cluster 3", "Percentage cluster 4", "Percentage cluster 5", "Percentage cluster 6", "Percentage cluster 7")) 

## Cell characterization cluster 6

cell_caract <-read.csv("./Results/clustering/cell_characterization_cluster6.csv", sep=";")
cell_caract_na <- subset(na.omit(cell_caract))
summary_df <- data.frame(Celularity=as.integer(as.vector(cell_caract_na$cellularity)))
summary_df[['Percentage of tumoral cells']] <- cell_caract_na$tumoral_cells_IHC0+cell_caract_na$tumoral_cells_IHC1+cell_caract_na$tumoral_cells_IHC2
summary_df[['Percentage of tumoral cells 0+']] <- as.integer(cell_caract_na$tumoral_cells_IHC0)
summary_df[['Percentage of tumoral cells 1+']] <- as.integer(cell_caract_na$tumoral_cells_IHC1)
summary_df[['Percentage of tumoral cells 2+']] <- as.numeric(cell_caract_na$tumoral_cells_IHC2)
summary_df[['Percentage of immune cells']] <- cell_caract_na$immune_cells
summary_df[['Percentage of fibroblast cells']] <- cell_caract_na$fibroblasts
summary_df[['Percentage of other cells']] <- cell_caract_na$other_cells
summary_df %>% tbl_summary(type = list("Percentage of tumoral cells 0+"	 ~ "continuous", "Percentage of tumoral cells 2+"	 ~ "continuous"))

summary_df = summary_df %>% tbl_summary(statistic =list(all_continuous() ~ "{mean} ({sd})"), 
                             type = list("Percentage of tumoral cells 0+"	 ~ "continuous", 
                                         "Percentage of tumoral cells 2+"	 ~ "continuous"), digits=everything() ~ 2) 
add_ci(summary_df)

caract_tum_cells <- subset(cell_caract, (tumoral_cells_IHC0!=0)|(tumoral_cells_IHC1!=0)|(tumoral_cells_IHC2!=0))
summary_df <- data.frame(a = (caract_tum_cells$tumoral_cells_IHC0 / (caract_tum_cells$tumoral_cells_IHC0+caract_tum_cells$tumoral_cells_IHC1+caract_tum_cells$tumoral_cells_IHC2))*100)
colnames(summary_df) <- c("Proportion of tumoral cells 0+ among all tumoral cells")
summary_df[["Proportion of tumoral cells 1+ among all tumoral cells"]] = (caract_tum_cells$tumoral_cells_IHC1 / (caract_tum_cells$tumoral_cells_IHC0+caract_tum_cells$tumoral_cells_IHC1+caract_tum_cells$tumoral_cells_IHC2))*100
summary_df[["Proportion of tumoral cells 2+ among all tumoral cells"]] = (caract_tum_cells$tumoral_cells_IHC2 / (caract_tum_cells$tumoral_cells_IHC0+caract_tum_cells$tumoral_cells_IHC1+caract_tum_cells$tumoral_cells_IHC2))*100
summary_df %>% tbl_summary(type=list("Proportion of tumoral cells 2+ among all tumoral cells" ~ "continuous"))

summary_df = summary_df %>% tbl_summary(statistic =list(all_continuous() ~ "{mean} ({sd})"), 
                             type =list("Proportion of tumoral cells 2+ among all tumoral cells" ~ "continuous"), digits=everything() ~ 3)
add_ci(summary_df)

## Percentage on pipeline figure

coul = rev(c("#E41A1C", "#377EB8",  "#4DAF4A", "#984EA3", "#FFFF33", "#A65628", "#F781BF", "#999999"))
names = rev(c("cluster 0", "cluster 1", "cluster 2", "cluster 3", "cluster 4", "cluster 5", "cluster 6", "cluster 7"))
val = rev(c(0.06925,0.05701,0.10748,0.13819,0.34691,0.07467,0.15667,0.04983))
barplot(height=val, names=names, col=coul, horiz=T, las=1, xlab = "Percentage")


## Nuclei analysis
library(forcats)
library(gtsummary)
data <- read.csv("./Results/nuclei/nuclei_features.csv")

table2 <- 
  tbl_summary(
    data[,c("n_nuclei","dab_max_mean" ,"cluster")],
    by = cluster, # split table by group
    missing = "no" # don't list missing data separately
  ) %>%
  add_n() %>% # add column with total number of non-missing observations
  modify_header(label = "**Variable**") %>% # update the column header
  bold_labels()
table2

data_na <- na.omit(data)

group_ordered <- with(data_na,                       # Order boxes by median
                      reorder(cluster,
                              -dab_max_mean,
                              median))

data_na$cluster <- factor(data_na$cluster,
                          levels = levels(group_ordered))
ggboxplot(data_na, x="cluster", y="dab_max_mean", fill="cluster", 
          outlier.shape = NA, palette="RdBu", xlab = "Cluster id", 
          ylab = "Maximum DAB intensity (optical density)",
          width = 0.3)  +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5), 
        axis.text=element_text(size=10), axis.title=element_text(size=12))

data_bis = data
data_bis[data_bis$n_nuclei>=4,]$n_nuclei = 4
tb=table(data_bis$n_nuclei,data_bis$cluster)
tb=as.data.frame(tb)

names(tb)=c("nuc","clu","frq")

vec <- tb %>%
  group_by(clu) %>%
  summarise(sum(frq))

tb$proportion <- round(tb$frq / rep(vec$`sum(frq)`,each=5), 3)

ggplot(data = tb[!tb$proportion == 0, ], aes(x = factor(clu, level=c(4,0,2,1,5,7,6,3)), y = proportion, fill = forcats::fct_rev(nuc))) + 
  geom_bar(stat = "identity", position = 'fill', colour="black") +   scale_fill_manual(
    values = rev(c('#fff7ec','#fdbb84','#fc8d59','#d7301f','#990000')),labels = c("> or = 4", "3", "2", "1", "0")
  ) + theme_light() +theme(axis.line = element_line(colour = "black"),
                           panel.grid.major = element_blank(),
                           panel.grid.minor = element_blank(),
                           panel.border = element_blank(),
                           panel.background = element_blank()) + labs(x="Cluster id",
                                                                      y="Proportion of patches (%)",
                                                                      fill="Number of nuclei by patch",
                           ) + scale_y_continuous(labels = scales::percent_format()) + theme(axis.text=element_text(size=10), axis.title=element_text(size=12))



# COHORT 2 analysis

## Optimal number of clusters

score_davies = read.csv("./Results/clustering/score_optimal_nb_clusters_davies_bouldin_cohort2.csv")

to_plot_davies = data.frame("Number_of_clusters"=c(7:12), "Davies_Bouldin_index"=score_davies$davies_bouldin_score)

ggline(to_plot_davies, x="Number_of_clusters",y="Davies_Bouldin_index", color="steelblue",size=1, ggtheme=theme_classic(),
       xlab="Number of clusters", ylab="Davies-Bouldin index", ylim=c(3.8,4.3)) +
  theme(axis.text=element_text(size=10), axis.title=element_text(size=12)) + 
  scale_x_continuous(breaks = to_plot_davies$Number_of_clusters) +
  geom_segment(aes(x=8, y=2.8, xend=8, yend=Davies_Bouldin_index[2]), linetype="dashed", size=1)

## Box plots of clusters' percentage

my_pairs = list(c(c('percentage_cluster0', 'Response'),
                  c('percentage_cluster0', 'Non-response')),
                c(c('percentage_cluster1', 'Response'),
                  c('percentage_cluster1', 'Non-response')),
                c(c('percentage_cluster2', 'Response'),
                  c('percentage_cluster2', 'Non-response')),
                c(c('percentage_cluster3', 'Response'),
                  c('percentage_cluster3', 'Non-response')),
                c(c('percentage_cluster4', 'Response'),
                  c('percentage_cluster4', 'Non-response')),
                c(c('percentage_cluster5', 'Response'),
                  c('percentage_cluster5', 'Non-response')),
                c(c('percentage_cluster6', 'Response'),
                  c('percentage_cluster6', 'Non-response')),
                c(c('percentage_cluster7', 'Response'),
                  c('percentage_cluster7', 'Non-response')))
                  
data_box_plots = read.csv("./Results/clustering/percentage_clusters_cohort_2.csv")
data_box_plots$Objective.Response <- mapvalues(data_box_plots$Objective.Response,
                                               from=c("Negative Response","Positive Response"),
                                               to=c("Non-response","Response"))

box_plot <-ggboxplot(data_box_plots, x = "Feature.names", y = "value",
                     color = "Objective.Response", palette = "npg",
                     add = "point", xlab="Cluster id", ylab="Cluster percentage") + labs(color='Objective response to treatment :') 

stat.test <- data_box_plots %>%
  group_by(Feature.names) %>%
  wilcox_test(value ~ Objective.Response)
stat.test$p.adj = format(round(p.adjust(stat.test$p, method = "BH", n = length(stat.test$p)), 4), nsmall=4)
stat.test <- stat.test %>% 
  add_xy_position(x = "Feature.names", dodge = 0.8)


box_plot + stat_pvalue_manual(stat.test, label="p = {p.adj}", tip.length = 0.01) + scale_x_discrete(labels = c("0", "1", "2", "3", "4", "5", "6", "7")) + scale_y_continuous(labels = scales::percent, breaks=c(0,.2,.4,.6,.8))

data_box_plots = read.csv("./Results/clustering/percentage_clusters_cohort_2_pretrained.csv")
data_box_plots$Objective.Response <- mapvalues(data_box_plots$Objective.Response,
                                               from=c("Negative Response","Positive Response"),
                                               to=c("Non-response","Response"))

box_plot <-ggboxplot(data_box_plots, x = "Feature.names", y = "value",
                     color = "Objective.Response", palette = "npg",
                     add = "point", xlab="Cluster id", ylab="Cluster percentage") + labs(color='Objective response to treatment :') 

stat.test <- data_box_plots %>%
  group_by(Feature.names) %>%
  wilcox_test(value ~ Objective.Response)
stat.test$p.adj = format(round(p.adjust(stat.test$p, method = "BH", n = length(stat.test$p)), 4), nsmall=4)
stat.test <- stat.test %>% 
  add_xy_position(x = "Feature.names", dodge = 0.8)

box_plot + stat_pvalue_manual(stat.test, label="p = {p.adj}", tip.length = 0.01) + scale_x_discrete(labels = c("0", "1", "2", "3", "4", "5", "6", "7")) + scale_y_continuous(labels = scales::percent, breaks=c(0,.2,.4,.6,.8), limits=c(0,.8)) 




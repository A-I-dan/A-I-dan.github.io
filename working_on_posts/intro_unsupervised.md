---
layout:     post
title:      "A Brief Introduction to Unsupervised Learning"
date:       2019-03-30 12:00:00
author:     "A.I. Dan"
---

# A Brief Introduction to Unsupervised Learning

<b>Summary:</b> In this post I will discuss the details of <b>unsupervised</b> machine learning and its applications. Code examples will be used for demonstration but the theory and background knowledge will be the main focus.

Unsupervised learning does not use labeled data, but instead focuses on the data's features. We are not concerned with the targeted outputs because making predictions is not the outcome of unsupervised learning algorithms.  

The goal of unsupervised learning algorithms is to analyze data and find important features. Unsupervised learning will often find subgroups within the dataset.

<hr>

### Clustering

Clustering aims to discover "clusters", or subgroups within the data. Clusters will contain data points that are similar to each other.  Clustering helps find underlying patterns within the data that may not be noticeable to a human observer.

### K-Means Clustering

In k-means clustering, the goal is to partition the data into a predetermined value for *K*, the number of clusters. Each data point will fall into only one cluster of the *K* clusters, and therefore the clusters will not overlap.  


### Hierarchical Clustering

Unlike k-means clustering, with hierarchical clustering, the number of clusters in unknown beforehand.

---
layout: post
title: "A Brief Introduction to Berserker"
date: 2015-11-01 18:00:00
author: Jake
categories:
- blog
img: none.png
thumb: bar.png
---

If you compete in or follow the machine learning competitions on [kaggle](http://kaggle.com), then by now you're probably familiar with the concept of ensembling.  Almost without exception, the winners of each competition are combining a variety of estimators into a single, more powerful model.  It doesn't matter what kind of data is given, or whether regression or classification is required, there isn't a single machine learning algorithm that can compete with an ensemble. Even *if* a shiny new algorithm were published tomorrow that was objectively better than random forests, gradient boosting, and the like - you could just add it to you ensemble, making it even better.

There exists a variety of ensembling techniques, which mostly stem two methods known as "stacking" and "blending".  I may go over these in depth in a future post, but for now all you need to know is that several diverse models separately make predictions, which are then combined into a single prediction.  The basic idea is that while all models have imperfections, most of the models are correct for a given prediction so each individual error has negligible effect.

#### The Case for Berserker

I think its fair to say that most of us don't rewrite the random forest algorithm in C code every time we want to use it. We have things like scikit-learn so we don't have to continuously reinvent the wheel. Yet for some reason we don't have a generalized, reusable tools for creating ensembles, despite the fact that everyone and their mother is using them.  *That* is why I created [Berserker](https://github.com/jpopham91/berserker).

You can get all of the details in the readme, but here are a few key features:

- A familiar scikit-learn api/syntax
- Prediction memoization
- Generate models algorithmically

#### A Simple Example

Finally, I'll leave you with the source code and output for a demo using the popular Boston housing prices dataset.  With only a few lines of code, you can create an ensemble (of ensembles) which outperforms the vanilla random forest and GBT in scikit-learn.

```python
from berserker.ensemble import Ensemble
from berserker.layers import Layer
from berserker.nodes import Node

model = Ensemble(X_trn, y_trn, mean_squared_error)

# base estimator pool
model.add_layer(folds=5)
model.add_node(RandomForestRegressor(50), name='50 Tree Random Forest')
model.add_node(GradientBoostingRegressor(n_estimators=250), name='250 Gradient Boosted Trees')

# meta-estimator
model.add_layer()
model.add_node(LinearRegression(), name='Lin Reg Meta Estimator')

preds = model.predict(X_tst)
```
<pre>
Level 1 Estimators (12 features)     Validation Error
-----------------------------------------------------
50 Tree RF                            16.1368
Gradient Boosted Trees                18.4357

Level 2 Estimators (14 features)      Validation Error
-----------------------------------------------------
<b>Lin Reg Meta Estimator                15.5071</b>
</pre>

I urge you to try it out.  This is my first attempt at writing a library, so I openly welcome any criticism.

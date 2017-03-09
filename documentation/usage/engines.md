---
title: Engines
layout: default
documentation: true
category_weight: 5
categories: [Usage]
---

As with so many things, there is no one-size-fits-all in probabilistic model checking. Generally, it depends on the characteristics of the input model, which of the model checking approaches works best (for example in terms of time or memory requirements). Therefore, established model checkers provide a range of different *engines* that pursue different approaches to solve the same problem. In practice, the functionality of the engines is typically incomparable, that is there may be features that an engine can solve that another one cannot and vice versa. In the following, we give a brief intuitive overview on the engines provided by Storm. We also indicate how to select them, in which cases certain engines are well-suited and which major restrictions exist.

## Sparse

Storm's main engine is the sparse engine in the sense that it tends to have the most features. It takes the model description and directly builds a representation based on *explicit data structures*, mainly [bit vectors](https://en.wikipedia.org/wiki/Bit_array) and [sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix). Then, model checking is performed using these data structures. Since these permit easy access to single elements, they are standard representations for many tasks involved in the solution procedure (like solving linear equations). This enables the use of off-the-shelf libraries, for instance [Eigen](http://eigen.tuxfamily.org/) or [gmm++](http://download.gna.org/getfem/html/homepage/gmm/), that implement sophisticated solution methods.

- able to use off-the-shelf libraries to solve equation systems (standard data structures), solving therefore tends to be fast
- well-suited for moderately sized models without a lot of symmetry

**Select**: `--engine sparse` or `-e sparse`

##### **Characteristics**:
- model building is rather memory- and time-consuming
- numerical solution tends to be fast

##### **Advisable if**:
- model is moderately sized
- model is asymmetrical
- model checking needs heavy numerical computations (i.e. many iterations, etc.)

##### **Major restrictions**:
- none

## DD

[Binary decision diagrams](https://en.wikipedia.org/wiki/Binary_decision_diagram) (BDDs) are a data structure to represent switching functions. They have proven to enable the verification of gigantic hardware circuits. Multi-terminal BDDs (MTBDDs) extend BDDs to allow representing functions that map to numbers rather than just true or false. The dd engine builds a representation of the model in terms of BDDs (state sets) and MTBDDs (matrices, vectors), which is often referred to as a *symbolic* representation (rather than an *explicit* representation as in the [sparse](#sparse) engine). As MTBDDs support certain arithmetical operations, they are also used in the quantitative model checking step. However, DDs do not allow for (efficient) random access, which limits the numerical solution techniques that are applicable.

- builds model in terms of decision diagrams
- model checking performed directly on these as they support arithmetical operations
- because of the peculiarities of the data structure, there is no (efficient) random access
- this means that this engine does not support many of the solution methods of the sparse engine
- in general, DDs are well-suited to succinctly represent gigantic symmetric systems
- numerical solutions involving a lot of iterations tend to be slower

**Select**: `--engine dd` or `-e dd`

##### **Characteristics**:
- model building is fast and memory-efficient if the model is somewhat symmetrical
- numerical solution tends to be slower

##### **Advisable if**:
- model is large
- model is symmetrical
- target query does not involve heavy numerical computations (for example qualitative queries)

##### **Major restrictions**:
- only supports [discrete-time models](models.html)

## Hybrid

The hybrid engine tries to combine the [sparse](#sparse) and [dd](#dd) engines. It first builds a DD-based representation of the system. Then, however, the model checking is performed in a hybrid manner. That is, operations that are deemed to be more efficient on DDs (qualitative computations) are carried out in this realm, while the heavy numerical computations are performed on an explicit representation derived from the systems' DDs.

- tries to combine the sparse and dd engines
- first builds the dd-based representation of the system like the dd engine
- however, the numerical computations of model checking tasks are performed on a sparse representation derived from the DDs
- well-suited if the model is large but still fits into memory in a sparse representation

**Select**: `--engine hybrid` or `-e hybrid`

##### **Characteristics**:
- model building is fast and memory-efficient if the model is somewhat symmetrical
- translation from DDs to sparse matrices requires the sparse matrix to fit into memory
- numerical solution tends to be fast

##### **Advisable if**:
- model is not too large
- model is symmetrical

##### **Major restrictions**:
- does not support [MAs](models.html#markov-automata)

## Exploration

All engines so far have the requirement that a representation of the model needs to be built prior to model checking. This can be prohibitive for some models and may be unnecessary as fragments of the model may already allow for an answer (within a given precision) to a query. The exploration engine is based on the ideas of applying techniques from *machine learning*[^1]. Intuitively, it tries to explore the parts of the system on-the-fly that contribute most to the model checking result. 

**Select**: `--engine expl` or `-e expl`

##### **Advisable if**:
- large but finite models
- small parts of the system potentially influence the model checking result significantly
- low target precision

##### **Major restrictions**:
- only supports [discrete-time models](models.html)
- only supports reachability objectives

## Abstraction-Refinement

All other engines are not suited for models with an infinite state space. The approach of the abstraction-refinement engine is to start with a coarse over-approximation of the concrete model. This *abstract model* is then analyzed. Based on the result, one of two things happen: either the result carries over to the concrete model and and an answer can be returned or the abstraction needs to be refined. In the latter case, the abstraction is analyzed again and the loop is repeated until a conclusive answer can be given.

{:.alert .alert-warning}
This engine relies heavily on SMT solving (more concretely an enumeration of all satisfying assignments of a formula) and Craig interpolation. Therefore, this engine needs [MathSAT](http://mathsat.fbk.eu/) and Storm has to be built with MathSAT support, which requires a [manual setup]({{ site.baseurl }}/documentation/installation/manual-configuration.html#mathsat).

**Select**: `--engine abs` or `-e abs`

##### **Advisable if**:
- model is gigantic or infinite
- model is well structured

##### **Major restrictions**:
- only supports [discrete-time models](models.html)
- only supports reachability objectives

[^1]: [Tomáš Brázdil, Krishnendu Chatterjee, Martin Chmelík, Vojtěch Forejt, Jan Křetínský, Marta Kwiatkowska, David Parker, Mateusz Ujma, *Verification of Markov Decision Processes Using Learning Algorithms*, 2014](https://link.springer.com/chapter/10.1007/978-3-319-11936-6_8)
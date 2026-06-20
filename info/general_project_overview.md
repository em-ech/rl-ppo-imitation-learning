# General Project Overview: Imitation Learning Pipeline

This project is a comprehensive, 5-phase pipeline designed to answer a central research question: **Can Imitation Learning be used to reduce the massive sample complexity required to train Reinforcement Learning (RL) agents?**

The pipeline explores this by training an expert agent, collecting its knowledge, training a student to mimic it, fixing the student's mistakes interactively, and finally using that student to warm-start a new RL training run. 

The project is structured into the following 5 phases:

## Phase 1: Expert Training (PPO)
* **Goal:** Train a flawless "Oracle" (an expert agent) that knows exactly how to walk perfectly.
* **Method:** We use **PPO** (Proximal Policy Optimization), a standard RL algorithm, to train an agent from scratch on both the `Walker2d` and `Ant` environments until it reaches high performance (the Walker scoring ~6043 and the Ant scoring ~6293).
* **Why?** You can't teach a student if you don't have a teacher. This expert will act as our teacher.

## Phase 2: Demonstrations
* **Goal:** Collect a "textbook" for the student to study.
* **Method:** We drop the perfectly trained Phase 1 Expert into the environment and record exactly what it sees (Observations) and what it does (Actions) for 100 complete episodes. We save these recordings as massive NumPy arrays.
* **Why?** This provides the static dataset we will use to train our student offline.

## Phase 3: Behavioural Cloning (BC)
* **Goal:** Train a "Student" agent using supervised learning.
* **Method:** We train a blank neural network to mimic the expert's behavior using the dataset from Phase 2. The student looks at an observation and tries to output the exact same action the expert took.
* **The Problem:** Behavioural Cloning suffers from **Covariate Shift** (compounding errors). If the student makes one tiny mistake in the live environment, it ends up in a weird posture it never saw in the "textbook." It panics, the errors compound, and it falls over.

## Phase 4: DAgger (Dataset Aggregation)
* **Goal:** Fix the compounding error problem of Phase 3.
* **Method:** DAgger is an *interactive* teaching method. We put the student in the live environment and let it try to walk. When it inevitably makes a mistake and gets into a weird posture, the Phase 1 Expert (the Oracle) watches and says, *"Here is how you recover from that specific mistake."* We add these new recovery lessons to the dataset and retrain the student.
* **Why?** By explicitly teaching the student how to recover from its own unique mistakes, DAgger solves the compounding error problem and makes the student incredibly robust.

## Phase 5: Imitation as PPO Pretraining
* **Goal:** The ultimate test—does all of this actually save time?
* **Method:** We take the brain of the BC or DAgger student and explicitly inject it into a brand new PPO agent to "warm-start" it. We then let this pre-trained PPO agent learn in the live environment for 1.5 million steps and compare it against a PPO agent learning from absolute scratch.
* **Why?** To prove that instead of wasting millions of steps randomly exploring, you can use imitation learning to inject expert knowledge into an agent upfront, massively accelerating the reinforcement learning process.

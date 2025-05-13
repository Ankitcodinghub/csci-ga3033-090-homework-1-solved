# csci-ga3033-090-homework-1-solved
**TO GET THIS SOLUTION VISIT:** [CSCI-GA3033-090 Homework 1 Solved](https://www.ankitcodinghub.com/product/csci-ga3033-090-homework-1-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;93508&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSCI-GA3033-090 Homework 1 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
<h2>Introduction</h2>
This homework is designed to follow up on the lecture about Deep Imitation Learning. For this assignment, you will need to know about the imitation learning algorithms we talked about in the class, in particular DAgger., If you have not already, we propose you brush up on the lecture notes.

&nbsp;

Code folder

Find the folder with the provided code in the following google drive folder: <a href="https://drive.google.com/drive/folders/1T8B3gSNWjQU-JpifHkEDm9FfoxJA6wB_?usp=sharing">https://drive.google.com/drive/folders/1T8B3gSNWjQU-JpifHkEDm9FfoxJA6wB_?usp=sharing</a>. Download all the files in the same directory, and then follow the instructions in the env_installation.md file, and then dagger_template.py file.

<h2>DAgger</h2>
In the class, we learned about the DAgger (dataset aggregation) algorithm, which is used to clone an expert policy. This method is quite useful especially when querying the expert is expensive, and thus we want to learn a policy that is almost as good as the expert without the high number of queries to it.

&nbsp;

In this homework, we have provided you with an environment that is hard to learn directly. Thankfully, we have access to an expert in this environment. <strong>In this homework, your task will be to utilize <em>DAgger </em>to learn a deep neural network policy that performs well on this task.</strong>

<strong>&nbsp;</strong>

<h3>Environment</h3>
The environment we will use in this homework is built upon the Reacher environment from OpenAI gym (<a href="https://gym.openai.com/envs/Reacher-v2/">https://gym.openai.com/envs/Reacher-v2/</a>). We have provided our environment in the reacher_env.py file in our code directory. It follows the OpenAI gym API, which you can learn more about at <a href="https://github.com/openai/gym#api">https://github.com/openai/gym#api</a>. For this homework, an agent in this environment is considered successful if it can achieve a mean reward of at least 15.0.

&nbsp;

In this homework, we will attempt to learn this agent from image observations. Unfortunately, learning this agent directly from images without any priors is incredibly difficult, since images can be from a very high dimensional space. Thankfully, we have access to an expert prediction for any state the environment is currently on, which can be retrieved by the get_expert_action() function call. <strong>Note: get_expert_action() does not take any arguments, thus you must be careful to call it right after you have called .reset() or .step() on the environment to get the associated expert action.</strong>

<h3>Question 1</h3>
Download the code folder, with every file associated, from here <a href="https://drive.google.com/drive/folders/1T8B3gSNWjQU-JpifHkEDm9FfoxJA6wB_?usp=sharing">https://drive.google.com/drive/folders/1T8B3gSNWjQU-JpifHkEDm9FfoxJA6wB_?usp=sharing</a> Complete the code template provided in dagger_template.py, with the right code in every TODO section, to implement DAgger. Attach the completed file in your submission.

<h3>Question 2</h3>
Create a plot with the number of expert queries on the X-axis, and the performance of the imitation model on the Y-axis. Elaborate if you see any clear trends here. (Hint: in the env, the variable expert_calls counts the number of expert queries.

<h3>Question 3</h3>
Could you potentially improve on the number of queries to the expert made by the DAgger algorithm? Think about when querying the expert may be redundant.

&nbsp;

<strong>Bonus points</strong>: Try implementing your answer from question 3, and generate a query-vs-reward plot similar to question 2 for this implementation. Compare this plot with your answer from Q2. Is there a clear improvement?

<h2>Python environment installation instructions</h2>
<ol>
<li>Make sure you have conda installed in your system. <a href="https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation">Instructions link here.</a></li>
<li>Then, get the conda_env.yml file, and from the same directory, run conda env create -f conda_env.yml. If you don’t have a GPU, you can remove the line saying – nvidia::cudatoolkit=11.1.</li>
<li>Activate the environment, conda activate hw1_dagger.</li>
<li>Then, install pybullet gym using the following instructions: <a href="https://github.com/benelot/pybullet-gym#installing-pybullet-gym">https://github.com/benelot/pybullet-gym#installing-pybullet-gym</a>

(New: alternately, just install pybullet-gym from here: <a href="https://github.com/shubhamjha97/pybullet-gym">https://github.com/shubhamjha97/pybullet-gym</a> thanks Shubham!)</li>
<li>If you installed it from the official repo, go to the pybullet-gym directory, find this file: pybullet-gym\pybulletgym\envs\roboschool\envs\env_bases.py and change L29-L33 to the following:

self._cam_dist = 0.75</li>
</ol>
self._cam_yaw = 0

self._cam_pitch = -90

self._render_width = 320

self._render_height = 240

<ol>
<li>If you are still having trouble with training, up the image resize from (60, 80) to something higher.</li>
<li>Finally, run the code with python dagger_template.py once you have completed all the to-do steps in the code itself.</li>
</ol>

Preface and acknowledgements
Preface for the first edition
My work in sequential decision problems grew out of research that
started in the 1980 s in trucking, and over my career spanned rail, energy,
health, finance, e-commerce, supply chain management, and even learning
for materials science. Sequential decision problems arise in daily activities
such as sports, cooking, shopping, and finding the best path toadestina-
tion. They also arise when designingaproduct forastartup, hiring people
for the startup, and designing marketing campaigns.
Theearlyworkinsequentialdecisionproblems(knownasdynamicpro-
grams or optimal control problems) focused on solvingafamous, and fa-
mously intractable, equation known as Bellman's equation (or Hamilton-
Jacobi equations for continuous problems). I joinedacommunity that
worked on methods for approximating these equations; this work produced
asuccessfulbookonapproximatedynamicprogramming,produingabreak-
through foraclass of resource allocation problems. Over time, however,
I came to realize that approximate dynamic programming wasapowerful
methodforsolvingaverynarrowrangeofproblems-theproverbialhammer
looking foranail.
My work onawide range of problems made me realize the importance
of usingabroad range of methods which could be found through the re-
search literature. I foundIcould model any sequential decision problem
with the same framework which involved searching over methods for mak-
ing decisions, generally known as "policies" in the research literature. I
was then able to organize the vast range of methods into four broad classes
(meta-classes) of policies which span any method for making decisions, in-
cluding anything proposed in the literature or used in practice (including
methods that have not been invented yet!).
Thisframeworkisthefoundationofagraduate-levelbookthat Ifinished
in 2022 called Reinforcement Learning and Stochastic Optimization: A uni-
fied framework for sequential decisions (see
https://tinyurl.com/R Land SO/). AsIwas writing this book, I realized
that sequential decision problems are universal, arising in every human ac-
i

ii Contents
tivity. Furthermore, these ideas could (and should) be taught toabroad
audience,andnotjustthetypical,analyticallysophisticatedcrowdthatwe
find in operations research, computer science, economics, and pockets of
engineering.
The goal of this book is to enable readers to understand how to ap-
proach, model and solveasequential decision problem, even if they are
never going to writealine of code. While this book is analytical, the real
goal is to teach readers how to think about sequential decision problems,
breaking them down into the five core elements ofasequential decision
model, modeling uncertainty, and then designing policies.
Just as there are many styles for teaching statistics within different
communities, I believe there will beasimilar evolution to teaching these
ideas to different audiences. The examples in this book come from oper-
ations research, whichIlike to call the mathematics of everyday life. I
think readers will find most of the examples to be familiar, independent
of their professional field. At the same time, I can easily see versions of
the book designed purely for different problem domains such as health, fi-
nance, energy, robotics, and supply chain management (and this is hardly
a comprehensive list).

Contents iii
Acknowledgments for the first edition
Any proper acknowledgment of the work behind this book would rec-
ognize everyone who contributed to the graduate-level text, Reinforcement
Learning and Stochastic Optimization: A unified framework for sequential
decisions. There are simply too many people to list them all here, and
I ask readers to check the Acknowledgments section in that book for my
best effort at recognizing the efforts of so many who contributed to my
understanding of sequential decision problems.
This said, I would like to recognizeafew people who contributed to
this book. First there was an enthusiastic group of interns who wrote all
the Python modules that are used in the exercises for this book: Raluca
Cobzaru, Andrei Grauer, Joy Hii, John Nguyen and Robert Raveaunu. I
am especially grateful to Dennis Djanka,aprofessorat Karlsruhe University
in Germany, who updated the original Python modules from Python 2 to
Python 3, and made revision that make the library easier to use.
SecondIwarmly acknowledge the efforts of Dr. Juliana Nascimento,
who went through every line of this Python code, fixing bugs, cleaning
the logic, and helping me write the problem sets that were based on these
exercises.
Finally and most important was my undergraduate class, ORF 411:
Sequential Decision Analytics and Modeling, that signed up for the course
and participated in the first course specifically on "sequential decision an-
alytics" taught anywhere. They helped me refine the lectures which can
befoundathttps://tinyurl.com/RLS Ocourses/(scrolldownto"Under-
graduate/masters course in sequential decision analytics" for the slides).
Warren B. Powell
Princeton, New Jersey
August, 2022

iv Contents
Preface for the second edition
In 2026 I made the decision to go down the path of publishing through
Kindle Direct Publishing,which Ichoseformynewmonographseries Bridg-
ing Decision Problems. When Isawhoweasyitwas, Irealizedthat Icould
do the same with Sequential Decision Analytics and Modeling. KDP will
make it possible for me to do minor updates along with new editions with-
out the overhead of working throughapublisher. It allows me to provide
a Kindle edition foraminimal price, along withamuch more reasonably
priced hardbound edition.
The second edition contains the same set of application chapters. The
biggest changes are in chapter 1,where Iincorporatedmyideasondefining
different types of decisions. Eachoftheapplicationchaptersnowstartwith
a "Chapter Overview" that helps readers to understand what the chapter
is about. The entire book also benefited fromamuch-needed proofreading
to fix minor edits and some occasional errors.
This edition also embracesaprocess thatIam calling "framing the
problem" which involves starting by identifying (in English) the perfor-
mancemetrics,thetypesofdecisionsbeingmade,andthesourcesofuncer-
tainty. My new monograph, (Powell 2026), addresses these three questions
over the course of 150 pages, so they are not as simple as they sound, even
without the mathematical modeling.
Each chapter now includesabrief section, right after the narrative,
called "Framing the Problem" that sets the stage for the section on math-
ematical modeling by listing the metrics, decisions and uncertainties. Our
application of framing will make the process seem much simpler than it is
formostrealproblems, since Idonotillustratetheprocessofstartingwith
a full list of metrics, decisions, and uncertainties which are then reduced to
those represented in the model.
Acknowledgments for the second edition
I would first like to acknowledge the many thousands of readers who
have downloaded this book. With this writing, the book has enjoyed close
to 18,000 downloads from around the world (see figure 1). The feedback
has been simply heartwarming.
An important feature of this book is the Python modules that accom-
panymostofthechapters. Afewyearsafterthefirsteditionwaspublished,
Ilearnedtomyconsiderabledisappointmentthat Pythonhadupdatedfrom
version 2 to version 3, and the original modules no longer worked (and I
gave up coding in 1990, a decision which was core to my success).
You can imagine my heartfelt gratitude when Dennis Djanka, a pro-
fessor at Karlsruhe University in Germany, reached out to me with the
information that he had completely rewritten the library in Python 3. In

Contents v
Figure 1: Geographicaldistributionofdownloadsoffirsteditionasof 2025
addition,hemadethefollowingadditions(ashesummarizeditinhisemail).
(cid:136) Introductionof abstract baseclasses SDP Modeland SDP Policy that
make it easy to setup new models and policies with minimum amount
of code.
(cid:136) Complete rewrite of the code for the Asset Selling, Medical Decision-
Diabetes and Stochastic Shortest Path static modules and created a
Jupyter Notebook for each of the problems that walks the user from
creatingamodel and policy to tuning policies and interpreting the
results.
Ipreviouslycreateda UR Lfor Dennis'versionofthedirectoryusinghttps:
//tinyurl.com/sdagithubnew/ while retaining my original directory at
https://tinyurl.com/sdagithub/. With the release of the 2 nd edition, I
have changed the original UR Lsothatitalsopointsto Dennis'newlibrary.
Warren B. Powell
Princeton, New Jersey
February, 2026

Chapter 1
Modeling sequential decision problems
The process of solving any physical problem (and in particular any sequen-
tial decision problem) on the computer requires buildingamathematical
model, as illustrated in figure 1.1. For decades, the research community
has usedastandard mathematical framework for decision problems where
all the data is known in advance (known as deterministic optimization). A
simple version ofadeterministic optimization problem, known asalinear
program, might be written
minc Tx, (1.1)
x
wherexisavector of elements that have to satisfyaset of constraints
which are typically written
Ax = b, (1.2)
x ≥ 0. (1.3)
It is not necessary to understand equations (1.1) - (1.3) (which requires
basic familiarity with linear algebra), but thousands of students graduate
eachyearfromcourseswheretheylearnthisnotation,andalsolearnhowto
translateawide range of physical problems into this notation. Then, there
aresoftwarepackagesthattranslateproblemsinthisformatintoasolution.
Most important, this notational language is spoken around the world. The
same statement can be made about statistical modeling/machine learning
which today isamuch larger community than the people who understand
equations (1.1) - (1.3).

2 1. Modeling sequential decision problems
Min E {Σ cx} Organize classlibraries, and set up
Ax = b communications and databases
x >0
Mathematician
Figure 1.1: The bridge between the real world and the computer isamathematical
model.
Wecannotmakethesamestatementaboutsequentialdecisionproblems
whichisaproblemclassthatisstudiedbyatleast 15 differentcommunities
usingeightfundamentallydifferentnotationalstyles,oftenusingmathemat-
ics that requires advanced training. Inthisbook,weuseateach-by-example
style to show how to model the incredibly rich class of problems that we call
sequential decision problems. While we focus on relatively simpler prob-
lems,ourframeworkcanbeusedtomodelanysequentialdecisionproblem.
In addition, the resulting model can be translated directly to software.
The analytical foundation of this book is contained in (Powell 2020)
(RLSO) which isagraduate level text centered on methodology. From
time to time we will refer to material in this book for readers who might be
interestedingreaterdepth,andweencouragetechnicallyinclinedreadersto
use RLSO asareference. However, it is not needed. This book is designed
to provide the contextual background in the form ofaseries of examples
that should enable readers to think clearly and precisely about sequential
decision problems, even if they will never writealine of code.
Thisbookisaimedatundergraduateormasterslevelstudentswhohave
takenacourseinprobabilityandstatistics(aknowledgeoflinearprogram-
ming is not necessary, although we have an example which requires solving
alinearprogram). Allthechaptersarebuiltaroundspecificexamples,with
theexceptionofchapter 1, which provides an overview of the entire model-
ingframework,andchapter 7,wherewepauseandusethefirstsixchapters
to illustrate some important principles.
The presentation should not require mathematics beyond what would
be expected inafirst course on probability and statistics. This said, the

1.1. Getting started 3
book is centered on showing how to describe sequential decision problems
using notation that is precise enough that it can be the basis of computer
software.
Python modules accompany most of the chapters; these modules were
written around the modeling framework that runs throughout the book. At
the same time, any software package that simulatesasequential decision
problem,regardlessofhowitisbeingsolved,canbetranslateddirectlyinto
the modeling framework we use. For this reason, we encourage readers to
look at any piece of notation asavariable inacomputer program.
1.1 Getting started
Sequential decision problems can always be written as
decision, information, decision, information, decision,...
Each time we makeadecision we incuracost or receiveacontribution or
reward(therearemanywaystomeasureperformance). Decisionsaremade
withamethod that we are going to refer to asapolicy. A major goal that
isacentral focus of this book is to design effective policies that work well
over time, in the presence of the uncertainty of information that has not
yet arrived.
Sequential decision problems are ubiquitous, arising in virtually every
human process. Table 1.1 providesasample list of fields, with examples of
some of the decisions that might arise. Most of these fields probably have
many different types of decisions, ranging in complexity from when to sell
anassetortoadoptanewwebdesign, tochoosingthebestdrug, material,
or facility to design, to managing complex supply chains or dispatching a
fleet of trucks.
Even more challenging than listing all the types of decisions is identi-
fying the different sources of uncertainty that arise in many applications.
Humanbehavior,markets,physicalprocesses,transportationnetworks,en-
ergy systems and the broad array of uncertainties that arise in health hint
at the diversity of different sources of uncertainty.
As this book is being written, humanity is struggling with the spread
ofvariationsof COVID-19. Dealingwiththispandemichasbeendescribed
as"mind-bogglinglycomplex," [USA Today, Sept 8, 2020]butthisisreally

4 1. Modeling sequential decision problems
Field Questions
Business Whatproductsshouldwesell,withwhatfeatures?
Whichsuppliesshouldyouuse? Whatpriceshould
you charge?
Economics What interest rate should the Federal Reserve
charge given the state of the economy? What lev-
els of market liquidity should be provided?
Finance What stocks shouldaportfolio invest in? How
shouldatrader hedgeacontract for potential
downside?
Internet What ads should we display to maximize ad-
clicks? Which movies attract the most attention?
When/how should mass notices be sent?
Engineering Howtodesigndevicesfromaerosolcanstoelectric
vehicles, bridges to transportation systems, tran-
sistors to computers?
Public health How should we run testing to estimate the pro-
gression ofadisease? How should vaccines be al-
located? Which population groups should be tar-
geted?
Medical research What molecular configuration will produce the
drug which kills the most cancer cells? What set
ofstepsarerequiredtoproducesingle-wallednan-
otubes?
Supply chain mgmt. Whenshouldweplaceanorderforinventoryfrom
China? Which supplier should be used?
Freight trans- Which driver should moveaload? What loads
portation shouldatruckloadcarriercommittomove? Where
should drivers be domiciled?
Information col- Where should we sendadrone to collect informa-
lection tion on wildfires or invasive species? What drug
should we test to combatadisease?
Multiagent systems How shouldalarge company in an oligopolistic
marketbidoncontracts,anticipatingtheresponse
of its competitors?
Algorithms What stepsize rule should we use inasearch al-
gorithm? How do we determine the next point to
evaluate an expensive function?
Table 1.1: Asampleofdifferentfieldsanddecisionsthatneedtobemadewithineach
field.
a byproduct ofafailure to think about the problem inastructured way.
We are going to show the reader how to break down problems intoaseries
of basic components that lead to practical solutions.
Our approach starts by identifying some core elements such as perfor-
mance metrics, decisions, and sources of uncertainty, which then leads to

1.1. Getting started 5
thecreationofamathematicalmodeloftheproblem. Thenextstepisusu-
ally (but not always) to implement the model on the computer, but there
are going to be many problems where the process of buildingacomputer
modelisimpracticalforanyofanumberofreasons. Forthisreason,weare
also going to consider problems where we have to test and evaluate ideas in
thefield. Toimproveperformance,weneedtofirstlearnhowtomakegood
decisions over time (this is how we control the system). Then, we turn to
the design of the system.
At this time, the academic community has not adoptedastandard
modeling process for sequential decision problems. This is in sharp con-
trast with the arena of static, deterministic optimization problems which
have followedastrict framework since the 1950 s (equations (1.1) - (1.3)
representasample of this framework). Our modeling process is based on
the presentation in (Powell 2020), which isabook aimed atatechnical au-
dience that is primarily interested in developing and implementing models
on the computer.
By contrast, this book is aimed atabroader audience that is first and
foremostinterestedinlearninghowtothinkaboutsequentialdecisionprob-
lems. It usesateach-by-example style that focuses on communicating the
modeling process which we feel can be useful even without ultimately cre-
ating computer models. Central to our approach is the creation ofamath-
ematical model that eliminates the ambiguity when describing problems
in plain English. For readers who are interested in developing computer
models, notation is the stepping stone to writing software. However, we
are going to primarily use mathematical notation to create clarity when
describingaproblem, even if the reader never intends to writealine of
code.
Our presentation proceeds as follows:
(cid:136) Chapter 1 providesalight introduction to the universal modeling
framework, illustrated using two inventory problems (a simple one,
andonethatisslightlymorecomplex),followedbyabriefdiscussion
ofmodelinguncertainty. Itthenprovidesanintroductiontothefour
classes of policies that cover every method for making decisions.
(cid:136) Chapters 2-6 each describeaspecific sequential decision problem to
illustrate the modeling framework usingateach-by-example style.
These applications were chosen to bring out each of the four classes
of policies.

6 1. Modeling sequential decision problems
(cid:136) Chapter 7 returns to the universal modeling framework in more de-
tail. A much more careful discussion is given of the four classes of
policiesaswellasdifferenttypesofstatevariables,usingtheexamples
in chapters 2-6 to provide context.
(cid:136) Chapters 8-14 provide additional examples, using more complex set-
tings to illustrate more advanced modeling concepts, covering both
uncertainty modeling (in particular the modeling of electricity prices
in chapter 8) andaricher set of policies.
The application chapters (2-6 and 8-14) all follow the same outline. They
can be covered in any order, keeping in mind that the applications in chap-
ters 2-6 are simpler and were chosen to illustrate each of the four classes of
policies. Readers interested in specific modeling topics (such as state vari-
ables, modeling uncertainty, or seeing different examples of policies) may
skim chapters, jumping directly to the topics that interest them.
Each chapter closes withaseries of exercises divided into three cate-
gories:
(cid:136) Review questions - These are simple questions that can be used to
reinforceabasic understanding from reading the chapter.
(cid:136) Problemsolvingquestions-Theseintroducemodelingchallengesthat
require problem solving skills.
(cid:136) Programmingquestions-Mostchaptershaveprogrammingexercises
that draw onaset of Python modules. The original Python mod-
ules,writtenin Python 2,benefitedfromamajorupgradeby Profes-
sor Dennis Djanka, a professor at Karlsruhe University in Germany.
The new library can be downloaded from https://tinyurl.com/
sdagithub/. Some of these questions require making programming
modifications to the Python code.
1.2 So, what isadecision?
Thereisalonghistory,datingbackover 2,000 yearstothedaysof Socrates,
Aristotle and Plato, documenting the study of how people make decisions.
Thenthereisasubstantialliterature,mostlysincethe 1950 s(butwithsome
important work before) on the mathematics of making optimal decisions,

1.2. So, what isadecision? 7
consisting of many thousands of papers and books. What this literature
seems to overlook is the basic question:
What isadecision?
We start with the observation thatadecision isaform of information
that affects the behavior of some "system" that we are looking to control.
Implicit in this system is one or more measures that quantify how well our
systemisperforming. Wethenhavetoidentifyanagentthatcontrolssome
aspect of our system.
Given this foundation, it helps to identify three classes of information:
1) The state of knowledge - This is information that we have right now
that is relevant to the performance of our system.
2) Informationthatchangesthestateofknowledgethatwecontrol(this
requires identifyingacontrolling agent for our system).
3) Information arriving to our system that changes the state of knowl-
edge that is beyond our control.
We refer to information in class 2 as decisions. This suggestsaformal
definition ofadecision as (Powell 2026):
Definition (formal): A decision is an endogenously controllable infor-
mation class.
An informal definition might be:
Definition (informal): A decision is something we control.
These definitions offerastarting point, but we do not learn very much
from them. More interesting in our view is to identify different types of
decisions. Below are six types of decisions that serve asastarting point:
1) Physical and financial decisions –Thesedecisionsariseintheman-
agement of physical and financial resources, such as people, equip-
ment,facilities,products,water,energy,aswellasfinancialresources
such as cash or investments. Decisions include buying, selling and
modifyingresources,whereamodificationmightmeanmovingitfrom
one location to another, repairing equipment, trainingaperson, or
combining ingredients to makeacake.
2) Discrete actions –Thisisageneraltermthatcanbeusedtodescribe
complexprojectssuchaslaunchinganewproduct,submittingadrug

8 1. Modeling sequential decision problems
toclinicaltrials,orpurchasingacompany. Discreteactionsmaymake
a number of different changes toasystem.
3) Information acquisition/observation decisions –Theseincludede-
cisionssuchasrunningexperimentsinthelab,fieldtests,orcomputer
simulations. It might include performing market research, hiring an
expert, or askingalarge language model.
4) Information communication/sharing decisions – These come in
two forms:
a) Messaging – This reflects what we say in text, video and/or
audio.
b) Channelsandtiming–Thiscoversthechoiceofhowtosendthe
information: text/ emails, publication (print or online), social
media, or advertising channels. It also requires choosing the
timing and frequency.
5) Choosing functions –Thesemaybemethodstomakedecisions(poli-
cies), the formulation of optimization models, the choice of perfor-
mance metrics, methods for forecasting or estimation, or the design
of transition functions (such as how disease spreads).
6) Setting parameters – There are oftenanumber of parameters that
affect the performance of asystem. Thesecould beprices, the coeffi-
cientsinastatisticalmodel,thetemperatureusedinamanufacturing
process. They might be the weight placed onaperformance metric,
or performance targets.
7) Estimation/identification - We may be givenapicture ofaperson,
and asked to identify them, or we may be givenaset of observations
of sales and asked to estimate future sales. In each case we have a
choice (of people, or of possible values of future sales) and we have
to decide which is best, minimizing some metric that describes the
error when we do not choose perfectly.
Implicit in the identification of decisions is understanding how the decision
affects the performance of the system. Moving physical resources (type
1) comes withacost, while satisfying demands brings revenues. A deci-
sion may have an immediate impact on one or more performance metrics
(as often occurs with managing resources), but often decisions have to be

1.3. Framing the problem 9
evaluated over time, and depend on information that is not known when
the decision is made. For this reason, we are often evaluating how we are
making decisions (that is, the method) as opposed to the decision itself.
1.3 Framing the problem
Thefirststepwhenapproachingadecisionprobleminvolvesansweringthree
questions:
(cid:136) What are the performance metrics?
(cid:136) What types of decisions are being made (and who makes them)?
(cid:136) What are the uncertainties that affect performance?
Note that the answers to these questions are fundamental to any decision
problem. In this book, these questions will seem fairly simple, because
we answer them in the context of the models that we have already design
to solveaproblem. In real applications, the lists of performance metrics,
decisions, and uncertainties can be quite long.
Asahint of the richness that framingaproblem can take on, we en-
courage the reader to look at the monograph Framing the Problem which
is dedicated to just this topic (Powell 2026). The monograph has entire
chapters dedicated to each of these questions, which are illustrated using a
dozen different applications.
The goal of the framing process is to identify what matters, starting
with the performance metrics, where evenasimple inventory problem can
be described with over 20 performance metrics, 30 different types of de-
cisions and over 30 types of uncertainties. The spreadsheet listing these
canbefoundathttps://tinyurl.com/Powell Inventory Decisions. This
does not mean that we will actually buildamodel with all this complexity.
For this reason, the book introducesadevice called interaction matrices
whereadomain expert prioritizes the metrics, and then uses judgment to
identify the decisions and uncertainties that have the largest impact on the
most important metrics.
Thisbookassumeswehavealreadyreducedaproblemtoasmallnum-
ber of metrics, decisions, and uncertainties, and uses these to focus the
development ofamathematical model.

10 1. Modeling sequential decision problems
1.4 The modeling process
Modeling is an art, but it is art guided byamathematical framework that
ensuresthatwegetawell-definedproblemthatwecanputonthecomputer
and solve. This can be viewed as buildingabridge fromamessy, poorly
defined real-world problem to something with the clarityacomputer can
understand, even if your end goal is not to put it on the computer.
Historically,ifamodelingeffortinvolvedtryingtomakedecisions,peo-
ple would turn to the well-known framework of deterministic optimization
whichoftenlookslikethemodelgivenbyequations(1.1)-(1.3)whichcon-
sists of decision variables x, an objective function cx, and the constraints
given by (1.3) - (1.3).
The problem with this classical modeling framework is what it leaves
out:
(cid:136) It assumes all the data (contained in the variables A, b and c) char-
acterizing the model is known perfectly.
(cid:136) Most decisions occur repeatedly over time, and yet there is no recog-
nition of this.
(cid:136) There is no way to represent the flow of information to the system.
(cid:136) Asabyproduct,theoptimalsolutiontothismodelcannotanticipate
events that affect the performance ofxin the field.
(cid:136) Thereisnowaytorepresentrisk,amajorissueinmanyapplications.
(cid:136) It assumesasingle decision-maker.
Mathematical models should, first and foremost, provideapath that
tells us how to think about problems. The classical deterministic optimiza-
tion models that follow the format of equations (1.1) - (1.3) completely
ignore anything related to the evolution of our problem over time.
This book is designed entirely aroundamodeling approach called the
universal modeling framework. Inanutshell, it aspires to represent any
aspect ofacontrollable system. Our default model will assume that the
system evolves over time, as new information arrives.
In this section, we are going to provideavery compact version of the
universal modeling framework. Then (in section 1.5) we are going to illus-
tratetheframework,initiallyusingaverysimpleinventoryproblem(insec-
tion 1.5.1) but then introducing some modest extensions (in section 1.5.2).

1.4. The modeling process 11
After presenting these examples, we are going to return in section 1.6 to a
more detailed presentation of the universal modeling framework.
1.4.1 A compact presentation ofadynamic model
We begin by observing that we can model any sequential decision problem
using the sequence
(S ,x ,W ,S ,x ,W ,...,S ,x ,W ,...,S ),
0 0 1 1 1 2 t t t+1 T
where:
(cid:136) S is the state variables that captures everything we need to:
t
a) Makeadecision at time t.
b) Compute the performance metrics at time t.
c) Any other information needed to compute (a) or (b) at any
point in the future.
ItisbesttothinkofSasthestateofinformationor,moregenerally,
t
the state of knowledge, at time t.
(cid:136) x represents decision variables which capture the elements that we
t
control, suchaswhethertosellahouse, thepaththroughanetwork,
thechoiceofdrugforatreatment,thepriceatwhichtosellaproduct,
or the choice of truck to moveaload of freight.
(cid:136) W is the information that arrives after we make the decision x ,
t+1 t
which might be the final selling price ofahouse, the travel times
throughanetwork, howapatient responds toadrug, the sales of a
product ataprice, and the loads of freight called in after we make
initialassignments. WeviewtheinformationinWascomingfrom
t+1
outside of our system, which means it is outside of our control. For
this reason, we refer to it as exogenous information.
TherearemanysettingswhereitisbesttothinkofWasafunction
t+1
W (S ,x )thatdependsonthecurrentstateSand/orthedecision
t+1 t t t
x . We discuss this in more detail in section 1.7. We are going to
t
useWasourdefaultnotation, but with the understanding that it
t+1
may be influenced by the stateSor decision x .
t t

12 1. Modeling sequential decision problems
The decisionxis determined by some method that we refer to as a
t
policy,whichwedenote Xπ(S ). Thenotationπ carriers information about
t
the structure of the function, which we represent byfinaset of potential
functions F, and any tunable parameters θ ∈ Θf, which is defined by
the structure of the function. Forexample,aninventorypolicymightbeto
orderθorder unitsanytimetheinventorygoesbelowθmin,whichmeansthe
tunable parameters are θ = (θorder,θmin). The structure of the function
would be one example ofafunction f.
Weassumewehaveatransitionfunctionthattakesasinputthestate S ,
t
decision x , and the exogenous informationWand gives us the updated
t t+1
state S . Transition functions areaset of equations that update each
t+1
element of the state variable S ,whichmighthavejustoneelement,ortens
t
of thousands (or much more).
We incuracontribution (or cost) C(S ,x ) when we make the decision
t t
x =Xπ(S )giventheinformationinstate S . Ourgoalistofindthepolicy
t t t
that maximizes some objective that depends on the contributions C(S ,x )
t t
where x = Xπ(S ). For more complex settings, C(S ,x ) can actually be
t t t t
a set of performance metrics, although we will need to combine them in a
way to identify which decisionxto choose.
t
Thisisaverycompactdescriptionofasequentialdecisionproblem. We
next describeaset of steps to follow in the modeling process.
1.4.2 The steps in the modeling process
It is possible to divide the entire modeling process into seven steps (for our
purposes). Preceding these steps (labeled below as "Step 0" isabrief sum-
mary of the technical complexity of the application to help guide readers.
Step 0. Chapter summary - We open each chapter withasummary of
what the chapter is going to cover and, in some cases, how it relates
to the material from other chapters. The summaries indicate what
approaches are used to modeling uncertainty and the policies that
are used.
Step 1. The narrative - This will beaplain English description of the
problem. The narrative will not provide all the information needed
to createamathematical model; rather, it isafirst step that should
give the modeler the big picture without getting lost in notation.
Step 2. Framingtheproblem-Thisconsistsofansweringthreequestions:

1.4. The modeling process 13
(cid:136) What are the performance metrics?
(cid:136) What types of decisions are being made (and in some cases,
which agent is making them)?
(cid:136) What are the sources of uncertainty that affect performance?
Step 3. Identifying the core elements of the problem, with special empha-
sis on three dimensions of any sequential decision problem. These
elements are described without using mathematics:
(cid:136) What metrics are we trying to impact? Individual fields (such
as supply chain management, health, energy, finance) will each
be characterized byanumber of metrics that might be de-
scribed using terms such as costs, revenues, profits, rewards,
gains, losses, performance and risk.
(cid:136) What decisions are being made? Identifying decisions is quite
easy for many problems such as computer games, but if we ad-
dressacomplex problem such as responding toapublic health
process, reducing the carbon footprint, or managingasupply
chain,thenidentifyingallthedecisionscanbequitechallenging.
(cid:136) What are the different sources of uncertainty? What are we
uncertain about before we start? What information arrives ex-
ogenously over time (that is, arrives after we makeadecision)?
Table 1.2 illustratesdifferentsourcesofuncertaintyforamodel
ofthedistributionof COVI Dvaccines(see(Powell 2020)[Chap-
ter 10] forapresentation of 12 classes of uncertainty).
Werefertotheprocessofansweringthesethreequestionsasframing
the problem.
Step 4. The mathematical model - Here we build off the first three ele-
mentsfrom Step 2, butnowwehavetocreateamathematicalmodel
that consists of five dimensions that apply to every sequential decision
problem:
(cid:136) State variables S - The state variable captures everything you
t
need to know at timetto makeadecision at time t, compute
costs and constraints, and if necessary, simulate your way to
time t+1. State variables can include information about phys-
ical resources (inventories or the location ofavehicle which en-
ter the problem through constraints), other information (such

14 1. Modeling sequential decision problems
Type of uncertainty Description
1) Observational errors Observing people with symptoms
Errorsclassifyingpeoplewithsymptoms
as having COVID
2) Exogenous uncertainty Reports of new cases, deaths
Availability of IC Us
Actual production of vaccines
3) Prognostic uncertainty Hospital admissions
Future performance of vaccines
Population response to vaccines
4) Inferential uncertainty Estimates of infection rates
Estimates of effectiveness of vaccines
5) Experimental uncertainty Drug performance inaclinical trial
Number being vaccinated
6) Model uncertainty Disease transmission rates
Geographical spread of infections
7) Transitional uncertainty Additions/withdrawals to/from vaccine
inventories
8) Control uncertainty Which population groups were vacci-
nated; vaccine allocations
9) Implementation uncertainty Failure to vaccinate
10) Communication errors Reporting errors from the field
Failure to notify when to be vaccinated
11) Goal uncertainty Disagreements in who should be vacci-
nated
12) Environmental uncertainty If/whenavaccine will be approved
Allocation of vaccines to different states,
countries
Table 1.2: Illustration of different types of uncertainty arising in the vaccination re-
sponsetothe COVI Dpandemic.
as costs or prices which enter the objective function), and be-
liefs about quantities and parameters we do not know perfectly
(such as forecasts or estimates of howapatient would respond
toadrug).
(cid:136) Decision variables x - These describe how we are going to de-
t
sign or control our system. Decisionshavetosatisfyconstraints
that we write as x ∈ X whereXcould beaset of discrete
t
choices, oraset of linear equations. Decisions will be deter-
minedbypoliciesthatarefunctions(orrules)thatwedesignate
by Xπ(S )thatdeterminex given what is in the state variable.
t t
Policies can be very simple (buy-low, sell-high) or quite com-

1.4. The modeling process 15
plex.
Theindexπ carries information about the type of function that
is used to make decisions, and any tunable parameters. Let
f ∈ F be the structure of the function, F be the set of pos-
sible functions, and let θ ∈ Θf be any tunable parameters for
functionf. Ourpolicywouldthenberepresentedasπ =(f,θ).
Section 1.8 givesanoverviewofthemajorfunctionclasses,each
of which has its own tunable parameters. We will often write
the policy as Xπ(S |θ) to indicate the dependence on tunable
t
parameters.
We return to this in considerable detail later in this chapter,
and throughout the book. Each of the examples given in the
book has been chosen to help illustrate specific types of policies.
(cid:136) Exogenous information W - This is new information that
t+1
arrives after we make decision x (but before we decide x )
t t+1
such as how much we sell after settingaprice, or the time to
complete the path we chose. Whenwemakeadecisionattimet,
theinformationinWisunknown,sowetreatitasarandom
t+1
variable when we are choosing x .
t
(cid:136) The transition function SM(S ,x ,W ) - These are the equa-
t t t+1
tions that describe how the state variables evolve over time.
For many real problems, transition functions capture all the
dynamics of the problem, and can be quite complex. In some
cases, we do not even know the equations, and have to depend
on just the state variables that we can observe. We write the
evolution of the state variablesSusing our transition function
t
as
S =SM(S ,x ,W ),
t+1 t t t+1
where SM(·)isknownasthestate(orsystem)transitionmodel
(hence theMin the superscript). The transition function de-
scribes how every element of the state variable changes given
the decisionsxand exogenous information W . In complex
t t+1
problems,thetransitionfunctionmayrequirethousandsoflines
of code to implement.
(cid:136) Theobjectivefunction-Thiscapturestheperformancemetrics

16 1. Modeling sequential decision problems
we use to evaluate our performance, and provides the basis for
searching over policies. We let
C(S ,x ) = the contribution (if maximizing) or cost
t t
(if minimizing) of decisionxwhich may
t
depend on the information in S .
t
In some settings it is more natural to write the single-period
contribution function as
C(S ,x ,W ) = the contribution function evaluated at the
t t t+1
end of time interval (t,t+1), after W
t+1
has been observed. For example, we may
order place an order forxthat arrive
t
right away to meet the uncertain demand
contained in W .
t+1
Ourobjectiveistofindthebestpolicy Xπ(S )tooptimizesome
t
metric such as:
– Maximize the expected sum of contributions over some
horizon.
– Maximize the expected performance ofafinal design that
we learned overanumber of experiments or observations.
– Minimize the risk associated withafinal design.
Our most common way of writing the objective function is
(cid:40) T (cid:41)
(cid:88)
max Fπ(S )=E C(S ,Xπ(S |θ))|S , (1.4)
0 t t 0
π=(f,θ)
t=0
where 'E" is called the expectation operator which means it
is taking an average over anything random, which might in-
clude uncertain information in the initial state S , as well as
the exogenous information process W ,...,W . It is standard
1 T
to write the expectation operator, but we can never actually
compute it. Later we show how to approximate it by running
aseriesofsimulationsandtakinganaverage, orbyobservinga
process in the field.
Caution has to be used when interpreting the expectation opera-

1.4. The modeling process 17
tor "E" in equation (1.4). What this operator literally means is
to "take an average over anything that is uncertain." The most
obvious piece of uncertainty is the exogenous information process
W ,W ,...,W ,...,W . Let ω beasingle sample path of the entire
1 2 t T
information process. We might constructasingle sample path from
history, in which case
The transition from the real problem (guided by the narrative) to
the elements of the mathematical model is perhaps the most difficult
step, as it often involves soliciting information fromanon-technical
source.
We note that we have presented the entire model without specifying
how we make decisions, which is represented by the policy Xπ(S ).
t
We call this "model first, then solve" and it representsamajor de-
parture from the vast literature that deals with sequential decision
problems. Itishardtocommunicatehowimportantitistoapproach
sequential decision problems in this way.
Step 5. The uncertainty model - This is how we model the different types
of uncertainty. There are two ways of introducing uncertainty into
our model:
1) Through the initial stateSwhich might specifyaprobabil-
ity distribution for uncertain parameters such as howapatient
might respond toadrug or how the market might respond to
price.
2) Through the exogenous information process W ,...,W .
1 T
We have three ways of modeling the exogenous information process:
(cid:136) Createamathematical model of W ,W ,...,W .
1 2 T
(cid:136) Use observations from history, such as past prices, sales, or
weather events.
(cid:136) Run the system in the field, observingWas they happen.
t
Step 6. Designing policies - Policies are functions, so we have to search
for the best function. (Yes, policies are functions to choose the best
decision, but choosing the policy is alsoadecision!) We do this by
identifying two core strategies for designing policies:

18 1. Modeling sequential decision problems
(cid:136) Search overafamily of functions to find the one that works
best, on average, over time.
(cid:136) Createapolicy by estimating the immediate cost or contribu-
tion ofadecision x , plus an approximation of future costs or
t
contributions,andthenfindingthechoicex thatoptimizesthe
t
sum of current and future costs or contributions. Google Maps
decides whether to turn left or right by optimizing over the time
required to transverse the next link in the network plus the re-
maining time to get to the destination. An inventory decision
might optimize over the cost of an order plus the estimated
valueofholdingacertainamountofinventorymovingforward.
We are going to be much more explicit about how to identify these
policies. Section 1.8 describes four classes of policies that will include
any method for making decisions (these are meta-classes).
Step 7. Evaluating policies - Finding the best policy means evaluating
policies to determine which is best. There are two ways to evaluate
a policy:
(cid:136) Testing the policy inacomputer simulator. This requires pro-
gramming all the equations in the state transition model
SM(S ,x ,W ) required to update the state variable S . It
t t t+1 t
also means being able to generate samples of W , which is
t+1
often the most subtle aspect ofasimulator.
(cid:136) Observing how the policy works in the field.
Simulatorscanbecomplexanddifficulttobuild,andarestillsubject
to modeling approximations. For this reason, the vast majority of
practical problems encountered in practice tend to involve testing in
thefield,whichisslow(ittakesadaytosimulateaday)andrequires
living with the results of the experiments.
The only way to become comfortable withamathematical model is
to see it illustrated usingafamiliar example. We start withauniversal
problem that we all encounter in everyday life: managing inventories.

1.5. Some inventory problems 19
1.5 Some inventory problems
We are going to illustrate our modeling framework using two variations
ofaclassic inventory problem, which is widely used as an application for
illustrating certain methods for solving sequential decision problems. We
start withasimple inventory example that gets across the core elements of
our modeling framework, but allows us to ignore many of the complexities
that we will be exploring in the remainder of the book.
Then, we are going to transition toaslightly more complicated in-
ventory problem that will allow us to illustrate some modeling principles.
Throughout the book, we are also going to use the idea of starting with a
basic version ofaproblem, and then introduce extensions that hint at the
types of complications that can arise in real applications.
1.5.1 A simple inventory problem
Oneofthemostfamiliarsequentialdecisionproblemsthatweallexperience
each time we visitastore is an inventory problem. We are going to use
a simple version of this problem to illustrate the six steps of our modeling
process that we introduced above:
Step 1: Narrative - A pizza restaurant has to decide how many pounds
of sausage to order from its food distributor. The restaurant has to
make the decision at the end of day t, communicate the order which
then arrives the following morning to meet tomorrow's orders. If
there is sausage left over, it can be held to the following day. The
costofthesausage,andthepricethatitwillbesoldforthenextday,
is known in advance, but the demand is not.
Step 2: The core elements of the problem are:
(cid:136) Metrics - We want to maximize profits given by the sales of
sausage minus the cost of purchasing the sausage.
(cid:136) Decisions - We have to decide how much to order at the end of
one day, arriving at the beginning of the next.
(cid:136) Sources of uncertainty - The only source of uncertainty in this
simple model is the demand for sausage the next day.
Step 3: - The mathematical model - This consists of five elements.

20 1. Modeling sequential decision problems
1) The state variable S - We distinguish between the initial state
t
variable S , and the dynamic state variableSfor t > 0. The
0 t
initial state variableSconsists of fixed parameters and initial
values of variables that change over time, giving us
S =(Rinv,(p,c),(D¯,σ¯D)).
0 0
We have divided the initial state into three types of variables:
(cid:136) Initial values of the resource state Rinv.
(cid:136) Values of constant parameterscand p.
(cid:136) Ourbeliefaboutthedemands, givenbyanormaldistribu-
tion with mean D¯ and standard deviation σ¯D.
The dynamic state variableSis our inventory which we are
t
going to call Rinv. For now, this is the only element of the
t
dynamic state variable, so
S =Rinv.
t t
Laterwearegoingtointroduceadditionalelementstoourstate
variable.
2) The decision variablexis how much we order at time t, which
t
we assume (fornow) arrives right away. We make ourdecisions
withapolicy Xπ(S ) which we design later.
t
3) Theexogenousinformationistherandomdemandforourproduct
which we are going to denote Dˆ , so W =Dˆ .
t+1 t+1 t+1
4) Our transition function captures how the inventoryRevolves
t
over time, which is given by
Rinv =max{0,Rinv+x −Dˆ }. (1.5)
t+1 t t t+1
5) Our objective function. For our inventory problem, it is most
natural to compute the contribution including the purchase cost
ofproductx and the revenue from satisfying the demand Dˆ ,
t t+1
whichmeansthatoursingle-periodcontributionfunctionwould
be written
C(S ,x ,Dˆ )=−cx +pmin{Rinv+x ,Dˆ },
t t t+1 t t t t+1

1.5. Some inventory problems 21
where x =Xπ(S ). Givenasequence of demands Dˆ ,...,Dˆ ,
t t 1 T
the value ofapolicy Fˆπ would be
T
Fˆπ(S )= (cid:88) C(S ,Xπ(S ),Dˆ ).
0 t t t+1
t=0
Our profits Fˆπ(S ) are random because it depends onapar-
ticular sequence of random demands Dˆ ,...,Dˆ . Finally we
1 T
average over these random demands by taking the expectation:
(cid:40) T (cid:41)
Fπ(S )=E (cid:88) C(S ,Xπ(S ),Dˆ )|S . (1.6)
0 t t t+1 0
t=0
Here,theconditioningontheinitialstateScanbereadassay-
ing "take the expectation given what we know initially." Con-
ditioningonSisimplicitanytimewetakeanexpectation,and
asaresult many authors leave it out. However, we are going
to include the conditioning onSto make it clear that if our
initial inputs (including beliefs) change, then this may have an
effect on howapolicy performs.
Step 4. The uncertainty model - The simplest approach to modeling un-
certainty is to just use historical data. The problem we might en-
counter is that if we run out of sausage, we might not observe the
full demand for sausage that day. If we are able to capture this lost
demand, then this isareasonable approach.
An alternative is to buildamathematical model. We might assume
that our demand is normally distributed with some mean D¯ and
standard deviation σ¯D. If we assume that both of these are known,
we can write our demand as
Dˆ ∼N(D¯,(σ¯D)2),
t+1
andtakeadvantageofpackagesthatcansamplefromthenormaldis-
tribution (for example, in Excel this is called
Norm.inv(Rand(),D¯,σ¯)togeneratearandomobservationwithmean
D¯ and standard deviation σ¯.
Using this model, we can createaset of demands (Dˆ ,Dˆ ,...,Dˆ ).
1 2 T
Then, we can repeat thisNtimes to createNsequences ofTde-

22 1. Modeling sequential decision problems
mands, giving us the sequence (Dˆn,Dˆn,...,Dˆn) for n = 1,...,N
1 2 T
that we need to estimate the value of the policy (we use this below
in Step 6).
Step 5. Designing policies - Next we have to designamethod for deter-
miningourorders. Acommonlyusedstrategyforinventoryproblems
is known as an "order-up-to" policy that looks like
(cid:40)
θmax−R if R <θmin,
Xπ(S |θ)= t t (1.7)
t
0 otherwise,
where θ =(θmin,θmax) isaset of parameters that need to be tuned.
It is called "order-up-to" since we place an order to bring the inven-
tory "up to" the upper limit θmax.
Step 6. Evaluating policies - There areavariety of strategies we might
use. Inpractice, we cannot compute the expectation in the objective
functioninequation(1.6),sowetakeaseriesofsamplesofdemands.
Let Dˆn,...,Dˆn be one sample of demands over t = 1,...,T, and
1 T
assume we can generateNof these. Now we can estimate our ex-
pected profits from policy Xπ(S ) by averaging over the samples for
t
n=1,...,N, which is computed using
N T
F π (θ|S )= 1 (cid:88)(cid:88) C(S ,Xπ(S |θ),Dˆn ).
0 N t t t+1
n=1 t=0
In plain English, we are simulating the policy Xπ(S |θ) N times us-
t
ing the simulated (or observed from history) samples of demands
Dˆn,...,Dˆn, and then averaging the performance to get F π (θ|S ).
1 T 0
We then face the problem of finding the best value of θ. A simple
strategy would be to generateKpossible values θ ,...,θ , simulat-
1 K
π
ing each one to find F (θ |S ) for each k, and then pick the value
k 0
of θ that works the best. This is not an optimal strategy, but it
k
providesasimple, practical starting point.
1.5.2 A slightly more complicated problem
The simple inventory problem above isaclassic setting for demonstrating
a particular method for solving sequential decision problems known as dy-
namic programming, which depends on havingasimple state variable that

1.5. Some inventory problems 23
isa)discreteandb)doesnothavetoomanypossiblevalues. Inourslightly
morecomplicatedinventoryproblem,wearegoingtoillustratethreediffer-
ent flavors of state variables which would representaserious complication
for one popular method for solving sequential decision problems, but has
no effect on the policy that we have chosen.
Step 1: Narrative - We again have our pizza restaurant that has to order
sausage,butwearegoingtoallowthepricewepayforsausagetovary
fromdaytoday,whereweassumethepriceononedayisindependent
of the price on the previous day. Then, we are also going to assume
thatwhilethedemandforsausagetomorrowisrandom,wearegoing
to be givenaforecast of tomorrow's demand that, while not perfect,
isbetterthannothavingaforecast. Otherwise,everythingaboutour
more complicated problem is the same as it was before.
Step 2: Core elements - These are:
(cid:136) Metrics - We want to maximize profits given by the sales of
sausage minus the cost of purchasing the sausage, where the
cost varies from day to day.
(cid:136) Decisions - As with our simpler inventory problem, we have to
decide how much to order at the end of one day, arrivingatthe
beginning of the next.
(cid:136) Sources of uncertainty - There are now three sources of uncer-
tainty: the difference between the actual demand and the fore-
cast, the evolution of the forecasts from one day to the next,
and the price we pay for the sausage.
Step 3: - Mathematical model - We still have the same five elements, but
now the problem isabit richer:
1) To construct the state variable, we need to list the information
(specifically, information that evolves over time) that is needed
in three different parts of the model:
1) The objective function.
2) The policy for making decisions (which includes the con-
straints).
3) The transition function.

24 1. Modeling sequential decision problems
Of course, we have not yet introduced any of these functions,
so you have to read forward, and verify that our state variable
contains all the information needed to compute each of these
functions. Think of this asadictionary of the information we
will need.
We start with the initial stateSwhich consists of constant
parameters,andinitialvaluesofquantitiesandparametersthat
change over time. These are:
(cid:136) Initial inventory - We start with an initial inventory R .
(cid:136) Initial purchase cost - c .
(cid:136) Price -We assume thatwe sell our sausageatafixedprice
p.
(cid:136) Initial forecast - We assume that our first forecast f D is
0,1
given, where f D is the forecast known at time 0 for the
0,1
demand at time 1.
(cid:136) Initial estimate of the standard deviation of the demand -
σ¯D.
(cid:136) Initial estimate of the standard deviation of the forecast -
σ¯f.
This means our initial state variable is
S =(R ,c ,p,f D ,σ¯D,σ¯f).
0 0 0 0,1 0 0
We then have the information that evolves over time which
makes up our dynamic state variable S :
t
(cid:136) Current inventory Rinv - The inventory for the beginning
t
of time interval (t,t+1).
(cid:136) Purchase cost c - This is the cost of sausage purchased at
t
timetwhich is given to us at time t.
(cid:136) Demandforecastf D -Thisistheforecastof Dˆ given
t,t+1 t+1
what we know at time t.
(cid:136) Current estimate of the standard deviation of the demand
- σ¯D.
t
(cid:136) Current estimate of the standard deviation of the forecast
- σ¯f.
t

1.5. Some inventory problems 25
Our dynamic state variable is then given by
S =(Rinv,c ,f D ,σ¯D,σ¯f).
t t t t,t+1 t t
2) The decision variablexis how much we order at time t, which
t
we assume (fornow) arrives right away. We make ourdecisions
withapolicy Xπ(S ) which we design later.
t
3) The exogenous information now consists of:
(cid:136) Purchase costs cˆ - This is the purchase cost of sausage
t+1
on day t+1 which is specified exogenously.
(cid:136) Forecasts - Each time period we are givenanew forecast.
Let εf be the change in the forecast between timetand
t+1
t+1.
(cid:136) Demands-Finally, weassumethattheactualdemandisa
random deviation from the forecast, which we could write
Dˆ =f D +εD .
t+1 t,t+1 t+1
Our complete set of exogenous information variables can now
be written
W = (cid:0) cˆ ,εf ,εD (cid:1) .
t+1 t+1 t+1 t+1
4) Transition function - This specifies how each of the (dynamic)
state variablesSevolve over time. We update our inventory
t
using:
Rinv = max{0,Rinv+x −Dˆ }. (1.8)
t+1 t t t+1
The demand is the forecasted demand plus the deviation εD
t+1
from the forecast, giving us the equation:
Dˆ = f D +εD . (1.9)
t+1 t,t+1 t+1
We assume that our forecast is updated using
f D = f D +εf . (1.10)
t+1,t+2 t,t+1 t+1
Next, we are going to adaptively estimate the variance in the

26 1. Modeling sequential decision problems
demand and the demand forecast:
(σ¯D )2 = (1−α)(σ¯D)2+α(f D −Dˆ )2, (1.11)
t+1 t t,t+1 t+1
(σ¯f )2 = (1−α)(σ¯f)2+α(f D −f D )2, (1.12)
t+1 t t,t+1 t+1,t+2
where 0<α<1 isasmoothing factor.
Finally, we update the costcwith the "observed cost" cˆ
t+1 t+1
which we write simply as
c =cˆ . (1.13)
t+1 t+1
Equation(1.13)isanexampleofastatevariablethatweobserve
ratherthancompute,aswedidwiththeinventory Rinv in(1.8).
t
Equation(1.8)issometimesreferredtoas"modelbased,"since
it reflects the physics of how inventories are updated, while
equation (1.13) is called "model free," since we do not make
any attempt at modeling the underlying process that produces
the change in costs.
Our transition function
S =SM(S ,x ,W )
t+1 t t t+1
consists of the equations (1.8) - (1.13).
5) Finally, our single period contribution function would now be
written
C(S ,x ,Dˆ )=−c x +pmin{R +x ,Dˆ },
t t t+1 t t t t t+1
where the only difference with the simpler inventory problem is
that the costcis now time-dependent c . We break from our
t
convention of writing the contribution as C(S ,x ) and allow it
t t
to include revenues from the demands Dˆ .
t+1
We now state our objective function formally as
(cid:40) T (cid:41)
max E (cid:88) C(S ,Xπ(S |θ),Dˆ )|S . (1.14)
t t t+1 0
π=(f,θ)
t=0
The optimization max means that we are searching over all
π
possible policies represented by (f,θ), which literally means

1.5. Some inventory problems 27
searching over all the different functions we might use to make
adecision. Theexamplesinthisbookaregoingtodemonstrate
how we are going to search over functions.
Recallthatabovewestatedthattheindexπcarriesinformation
about the type of function f ∈F, and any tunable parameters
θ ∈ Θf. In practice, the search over the types of functions
f ∈ F tends to be ad hoc (a knowledgeable analyst chooses
functions that make sense foraproblem), whereasacomputer
algorithm performs the search for the best value of θ ∈Θf.
Section 1.8 helps to guide the process of designing policies in
more detail. In fact, a substantial portion of this book is dedi-
cated to illustrating different types of policies in the context of
different applications.
Step 4. The uncertainty model - We are going to assume that the exoge-
nous changes εD and εf are described by normal distributions
t+1 t+1
with mean 0 and variances (σ¯D)2 and (σ¯f)2, which we express by
t t
writing
εD ∼N(0,(σ¯D)2),
t t
εf ∼N(0,(σ¯f)2).
t t
Uncertainty models can become quite complex, but this will serve as
an illustration.
Step 5. Designing policies - Next we have to designamethod for deter-
mining our orders. Instead of the order-up-to policy of our simpler
model, we are going to suggest the idea of ordering enough to meet
the expected demand for tomorrow, with an adjustment. We could
write this as
Xπ(S |θ)=max{0,f D −R }+θ. (1.15)
t t,t+1 t
If we hadaperfect forecast, then all we have to order would be
f D (our forecast of Dˆ ) minus the on-hand inventory. However,
t,t+1 t+1
because of uncertainty we are going to add an adjustment θ so that
we have some buffer to avoid stockouts.
Step 6. Evaluating policies - This time we have to generate samples of
all the random variables in the sequence W ,W ,...,W . Again we
1 2 T

28 1. Modeling sequential decision problems
might generateNsamples of the entire sequence so we can estimate
the performance ofapolicy using
N T
F π (θ)= 1 (cid:88)(cid:88) C(S ,Xπ(S |θ),Dˆn ).
N t t t+1
n=1 t=0
We again face the problem of finding the best value of θ, but we
return to that challenge later.
1.6 The universal modeling framework
We are now ready to describe in more detail the elements of the univer-
sal modeling framework (UMF). We note that the UMF can model any
sequential decision problem. This fairly broad claim will become apparent
as the elements unfold, since we are just applying notation to the general
statement ofasequential decision problem.
1.6.1 The five elements of the UMF
The UMF consists of the following elements.
1) The state variables S .
t
2) The decision variables x .
t
3) The exogenous information process W .
t
4) The state transition model SM(S ,x ,W ).
t t t+1
5) The objective function.
We describe these in more detail as follows:
State variables -ThestateSofthesystemattimethasalltheinforma-
t
tion that is necessary and sufficient to model our system from time t
onward. More specifically, this information consists of:
a) The information needed to makeadecision at time t.
b) Theinformationneededtocomputetheperformancemetricsat
time t.
c) Any information needed now to compute (a) and (b) in the
future.

1.6. The universal modeling framework 29
There are three types of information in S :
t
(cid:136) The physical state, R , captures physical quantities such as in-
t
ventories, people, available machines, facilities, water, drugs,
energy and money (in its various forms). R will also include
t
customer requests for products or services. In many applica-
tionsRdescribes the resource that is being managed, and a
t
fairly common error is to equate "state" with "physical state."
(cid:136) The information state, I , which contains the functions being
t
used (when there isachoice) and any tunable parameters. I
t
mightspecifyhowweareforecastingdemands,andtheparame-
tersusedtofittheforecast,inadditiontoanyotherparameters
that control the evolution of the system.
(cid:136) The belief state, B , which contains estimates or beliefs about
t
quantities and parameters that are not known perfectly. Thus,
B could capture the estimated mean and variance ofanormal
t
distribution(aswithourdemandforecastabove). Alternatively,
it could beavector of probabilities that evolve over time.
The physical stateRmight be the amount of money inacash ac-
t
count, whileImight be the current state of the stock and bond
t
markets. If we are traveling overadynamic network, R might be
t
our location on the network, whileIcould be what we know about
t
the travel times over each link. If we planapath and then wish to
penalize deviations from the plan, then the plan would be included
in the state variable through I .
t
State variables typically are not obvious. They emerge during the
modeling process, rather than something you can just immediately
write out. Just because we write it first does not mean that you will
always be able to list all the elements of the state variable right away.
But in the end, this is where you store all the information you need
to model your system from timetonward.
Decision variables -Differentcommunitiesusedifferentnotationsforde-
cision,suchasa fora(typicallydiscrete)actionoru fora(typically
t t
continuous) control in engineering. We usexas our default since it
t
is widely used by the math programming community.
Decision variables come in different flavors:

30 1. Modeling sequential decision problems
(cid:136) Binary(e.g. formodelingwhetherto sellanassetor not, or for
A/B testing of different web designs).
(cid:136) Discrete (e.g. choice of drug, which product to advertise).
(cid:136) Continuous scalar (prices, temperatures, concentrations).
(cid:136) Vectors(discreteorcontinuoussuchasallocationsofbloodsup-
plies among hospitals).
(cid:136) Categorical(e.g. whatfeaturestohighlightinaproductadver-
tisement).
Wenotethatthereareclassesofalgorithmsdeterminedbythenature
of the decision variable.
We assume that decisions are made withapolicy, which we might
denote Xπ(S )ifweusex asourdecision. Weassumethatadecision
t t
x = Xπ(S ) is feasible at time t, which means x ∈ X for some set
t t t t
(or region) X , which may depend on S .
t t
We let "π" carry the information about the type of function f ∈ F
(forexample,alinearmodelwithspecificexplanatoryvariables),and
any tunable parameters θ ∈Θf.
Exogenous information -WeletWbe any new information that first
t+1
becomes known at time t+1 (that is, betweentand t+1), where
the source of the information is from outside of our system (which
is why it is "exogenous"). When modeling specific variables, we use
"hats" to indicate exogenous information. Thus, Dˆ could be the
t+1
demand that arises betweentand t+1, or we could let pˆ be the
t+1
change in the price betweentand t+1.
Theexogenousinformationprocessmaybestationaryornonstation-
ary,purelyexogenousorstate(andpossiblyaction)dependent(ifwe
decide to sellalot of stock, it could push prices down).
We let ω representasample path W ,...,W , which represents a
1 T
sequence of outcomes of each W . Often, we will createaset Ω
t
of discrete samples, where each sample representsaparticular se-
quence of the outcomes of ourWprocess which we could write as
t
W (ω),...,W (ω). Ifwehave 20 samplepaths, wecanthinkofω as
1 T
consisting ofanumber between 1 and 20, which allows us to look up
the sample path.

1.6. The universal modeling framework 31
Transition function - We denote the transition function by
S =SM(S ,x ,W ), (1.16)
t+1 t t t+1
where SM(·) is also known by names such as state transition model,
systemmodel,plantmodel,plantequation,stateequation,andtrans-
fer function.
Equation (1.16) is the classical form ofatransition function which
gives the equations from the stateStothestate S . Equation(1.5)
t t+1
was the only transition equation for our simple inventory example,
whileequations(1.8)-(1.13)madeupthetransitionfunctionforour
more complicated example.
The transition function might capture any of the following types of
updates:
(cid:136) Changesinphysicalresourcessuchasaddinginventory,moving
people, or modifying equipment.
(cid:136) Updates to information such as changes in prices and weather.
(cid:136) Updatestoourbeliefsaboutuncertainquantitiesorparameters.
Thetransitionfunctionmaybeaknownsetofequations,orunknown,
such as when we describe human behavior or the evolution of CO 2
in the atmosphere. When the equations are unknown the problem
is often described as "model free" or "data driven" which means we
can only observe changes inavariable, rather than usingaphysical
model. Equation (1.13), where we "observe" the cost c = cˆ ,
t+1 t+1
with no idea how we evolved from c , is an example ofamodel free
t
transition.
Transitionfunctionsmaybelinear,continuousnonlinearorstepfunc-
tions. When the stateSincludesabelief state B , then the tran-
t t
sition function has to include the updating equations (we illustrate
this later in the book).
Givenapolicy Xπ(S ), an exogenous processWandatransition
t t+1
function,wecanwriteoursequenceofstates,decisions,andinforma-
tion as
(S ,x ,W ,S ,x ,W ,...,x ,W ,S ).
0 0 1 1 1 2 T−1 T T

32 1. Modeling sequential decision problems
Objective functions - There areanumber of ways to write objective
functions. One of the most common, which we will use asadefault,
maximizestotalexpectedcontributionsoversomehorizont=0,...T
(cid:40) T (cid:41)
(cid:88)
max Fπ(S )=E C (S ,Xπ(S |θ))|S , (1.17)
0 t t t t 0
π=(f,θ)
t=0
where
S =SM(S ,Xπ(S ),W ). (1.18)
t+1 t t t t+1
The model is fully specified when we also haveamodel of the initial
state S ,andamodeloftheexogenousprocess W ,W ,.... Wewrite
0 1 2
all the exogenous information as
(S ,W ,W ,...,W ). (1.19)
0 1 2 T
Equations(1.17),(1.18)and(1.19)constituteamodelofasequential
decision problem.
Moving forward, for compactness we are going to use max to repre-
π
sentasearch over the types of functions f ∈F and tunable parame-
ters θ ∈Θf.
Equation (1.17) uses an expectationEwhich means to take an av-
erage over all the possible outcomes of W ,...,W . This is virtu-
1 T
ally never possible to do computationally. Instead, let ω represent
a single outcome of the sequence W ,...,W which we might write
1 T
W (ω),...,W (ω). Assume that we can createNpossible outcomes
1 T
ofthissequence,andletωn represent how we index then th sequence.
If we are followingasample path ω, we would then rewrite our tran-
sition function in (1.18) using
S (ω)=SM(S (ω),Xπ(S (ω)),W (ω)). (1.20)
t+1 t t t t+1
We index every variable in equation (1.20) by ω to indicate that we
are followingasingle sample path of values of W .
t
Wecannowreplaceourexpectation-basedobjectivewithanaverage

1.6. The universal modeling framework 33
which we can write
N T
max F π (S )= 1 (cid:88)(cid:88) C (S (ωn),Xπ(S (ωn))). (1.21)
π 0 N t t t t
n=1 t=0
Often, we are just working withasingle sample path, possibly from
history. In this case, we are approximating the performance of the
policy using this single sample path, which we can write as
T
max Fˆπ(ω|S )= (cid:88) C (S (ω),Xπ(S (ω))). (1.22)
0 t t t t
π
t=0
Any time we write an objective using an expectation as in (1.17),
remember that what we would really do is to use an average as we
do in (1.21) orasample as in (1.22).
The expectation may also need to reflect uncertainty in the initial
state S , which might capture beliefs about uncertain forecasts, or
uncertain estimates about the state of disease inapatient. In this
case, the sample path ω needs to include samples from these initial
distributions.
Therewillbesomesettingswhereitmakesmoresensetouseacounter
nratherthantime. Inthiscase,welet Sn bethestateafternobservations
(these may be experiments, customer arrives, iterations of an algorithm).
We will use timetas our default index.
1.6.2 The initial state variables S
We need to distinguish between the initial stateSand subsequent states
S for t>0:
t
S - The initial stateScaptures i) deterministic parameters that never
0 0
change, ii) initial values of quantities or parameters that do change
(possiblyduetodecisions),andiii)beliefsaboutquantitiesorparam-
eters that we do not know perfectly (this might be the parameters
ofaprobability distribution) such as how we respond toavaccine or
how the market will respond to price. Thebeliefsmayremainstatic,
or we may updated them as we learn from observations.
S - This is all the information we need at timetfrom history to model
t
the system from timetonward. S for t > 0 only includes variables
t

34 1. Modeling sequential decision problems
that are changing overtime, whichmeansthatattimetwemayalso
be using static information contained in S .
We write the explicit dependence of the performance of the policy on
the initial state S , whether we use Fπ(S ), F π (S ) or Fˆπ(ω|S ). While
0 0 0 0
this should be obvious, it is often overlooked. The initial state includes
elements such as:
(cid:136) Initial values of the quantities of physical or financial resources R -
Thismightbestartinginventories,theinitiallocationofavehicle,the
available machines, and the initial set of facilities. It also includes
any static values, such asatransportation network, the size of a
warehouse (which does not change), and the number of trucks in a
fleet.
(cid:136) Initial values of parameters, along with any functions used to model
the problem I - This could be an initial price, the level of medica-
tion inapatient, along with the choice of functions for performing
forecasting or model the evolution of disease inapopulation.
(cid:136) Initial beliefs or estimates of any quantity or parameter B - This
could beademand forecast, the estimate of how markets respond
to prices, howapresidential candidate is polling, or beliefs in the
performance ofamanufacturing process.
We note it helps to separate initial values that never change, from those
that evolve over time, either directly asaresult of decisions or from exoge-
nous information. Values that never change are stored in S , but are not
represented inSfor t > 0. The reason for this is the desire to keepSas
t t
compact as possible.
Assume our policy Xπ(S |θ) has tunable parameters. For example, we
t
might be managing an inventory system where we use the familiar "order-
up-to" policy (known in the inventory literature as an (s,S) policy) given
by
(cid:40)
θmax−R R <θmin,
Xπ(S |θ)= t t
t
0 Otherwise.
where θ =(θmin,θmax). For simplicity we might assume when we place an
order it arrives right away (a standard textbook assumption that is never
trueinpractice)whichallowsustowritetheevolutionofourphysicalstate

1.6. The universal modeling framework 35
R (the amount in inventory just before we place our instantaneous order)
t
using
R =max{0,R +x −Dˆ }
t+1 t t t+1
where x = Xπ(S |θ) and Dˆ is the demand for our product over the
t t t+1
interval (t,t + 1) (this is our exogenous information W ). Finally let
t+1
C(S ,x ,W ) be our net profit over the interval (t,t+1) (which is not
t t t+1
important right now).
Nowimaginethatwehaveahistoricaldemandprocess W ,W ,...,W ,
1 2 t
...,W that allows us to runasimulation of our system. Let ω represent
T
this historical sequence of demands (or any exogenous information). We
would write the problem of finding the best set of ordering parameters θ
using
T
max Fˆπ(ω,θ|S )= (cid:88) C (S (ω),Xπ(S (ω))), (1.23)
0 t t t t
θ
t=0
where the state variable evolves according to
S (ω)=SM(S (ω),Xπ(S (ω)),W (ω)).
t+1 t t t t+1
Let θ∗ be the value of θ that we found by optimizing (1.23). The proper
way to write this optimal value is asafunction θ∗(S ) that depends on the
information in S (it also depends on the sample path ω). This helps to
communicate the reality that if we change the input data to our problem,
represented by S , then this may have an impact on the best values of our
policy parameters θ. In fact, we might even have to change our choice of
policy!
1.6.3 Variations
There are two important variations of our basic mathematical model:
(cid:136) From timetto iteration n - There are problem settings where it
is more natural to useacounternthan time t. We do more than
just change tt on since we view variables that change with iterations
differently from an evolution overtime. Specifically,weputtheindex
n in the superscript, such as Sn, xn and Wn+1.

36 1. Modeling sequential decision problems
One reason for this is that we viewaset of variables over time
x ,x ,...,x ,...,x asavector x = (x ,x ,...,x ,...,x ), which
1 2 t T 1 2 t T
is useful when modeling deterministic problems (we might optimize
over the entire vectorxat once). By contrast, we view xn as a
function that is evolving over time.
More practically, puttingnin the superscript allows us to write it-
erative simulations. So, we would write the information process over
time for iterationnusing
ωn =(Wn,...,Wn,...,Wn).
1 t T
If we are iteratively searching for the best policy, we might write our
policy for iterationnusing Xπ,n(S ), which then produces
t
Sn,xn,Wn,...,Sn,xn,WN ,...,SN,
0 0 1 t t t+1 T
where xn =Xπ,n(Sn|θ).
t t
(cid:136) Optimizing final reward - A common setting is where we are per-
formingstochasticsearch,aswouldhappenwhenlookingforthebest
policy. Each iteration for evaluating the algorithm might require a
simulation over time, although this is not always the case.
Now let's assume that our decision variable is the parameter θ, and
that we have an algorithm Θπ(Sθ,n) that works just likeapolicy
Xπ(S |θ), but where Sθ,n captures the "state" of the algorithm at
t
the nth iteration.
Search algorithms are all sequential decision problems, but we un-
like most sequential decision problems over time, we want to run N
iterations, and we only care about our solution at the end. Let
θπ,N = Thevalueofθn afterNiterations,whilefollowing"al-
gorithm" (policy) π.
The value θπ,N depends on the specific sequence of our information
process W 1,...,Wt,...,WN, but we then have to evaluate it using
a new set of samples that we are going to call W(cid:99).
We evaluate the performance of our algorithm usingafinal reward

1.7. Modeling uncertainty 37
objective, which we write as
max Fˆπ(Sθ) = E F(θπ,N,W(cid:99)) (1.24)
π 0 W(cid:99)
M
1 (cid:88)
≈ F(θπ,N,W(cid:99) m). (1.25)
M
m=1
Statedsimply,weevaluateourlearningpolicyforθ,whichwedenoted
Θπ(Sθ,N) by simulating throughNiterations using observations of
Wn(whichmaybeanentiresimulationovertimet). Whenweobtain
ourfinalestimateoftheparameterθ,whichwecallθπ,N,weevaluate
the performance of this value usingaseparate simulation where we
fix θ = θπ,N and then createanew set of random observations that
we call W(cid:99)m for m=1,...,M.
1.7 Modeling uncertainty
For many complex problems (supply chains, energy systems, and public
health are justafew), identifying and modeling the different forms of un-
certainty can bearich and complex exercise. We are going to hint at the
issues that arise, but we are not going to attemptathorough discussion of
this dimension.
Uncertainty is communicated to our model through two mechanisms:
the initial state S , which is where we would model the parameters of
probability distributions describing quantities and parameters that we do
not know perfectly, and the exogenous information process W ,...,W .
1 T
1.7.1 The initial state variables S
The initial state variable may contain deterministic parameters or initial
values of dynamically varying quantities and parameters. If this is all that
is in the initial state, then it is not capturing any form of uncertainty.
There are many problems where we do not know some quantities or
parameters, but can represent what we do know through the parameters of
a probability distribution. Some examples are:
(cid:136) The response ofapatient toanew medication.
(cid:136) Howamarket will respond toachange in price.

38 1. Modeling sequential decision problems
(cid:136) Howmanysalableheadsoflettucewehaveininventory(anuncertain
number may have wilted and are no longer salable).
(cid:136) The time at whichabox of inventory previously ordered from China
will arrive.
(cid:136) The amount of deposits toamutual fund vary randomly around a
mean λ, but we do not know what λ is.
Theseareanumberofwaysthatwecaninitializeamodelwithuncertainty
in some of the inputs.
An initial probabilistic belief may come from subjective judgment, or
from previous observations or experiments.
1.7.2 The exogenous information process
The second way uncertainty enters our model is through the exogenous in-
formationprocess. ThevariableWcontains information that is not known
t
until time period t. This means we have to makeadecisionxat time t
t
before we know the outcome of W .
t+1
Below isa listof examplesofWthatare revealed aftera decisionx
t+1 t
is made:
(cid:136) We chooseapath, and then observe the travel time on the path.
(cid:136) We chooseadrug, and then observe how the patient responds.
(cid:136) We chooseacatalyst, and then observe the strength of the material
that it produces.
(cid:136) We chooseaproduct to advertise in an online market, and then
observe the sales.
(cid:136) We selectaweb interface design, and then observe the number of
clicks that it can generate.
(cid:136) We allocate funds into an investment, and then observe the change
in the price of the investment.
In each case, the information we observe after we make the decision affects
theperformanceofthedecision(andwhichdecisionwouldhavebeenbest).
By now the reader has probably realized thatWis usuallyacollec-
t+1
tion of different types of information. For example, imagine that we are

1.7. Modeling uncertainty 39
treatingapatient that is experiencing elevated blood sugar. The physi-
cian wants to experiment with different strategies, ranging from diet and
exerciseordrugstoreduceweight,upthroughmedicationsthatspecifically
target blood sugar. The sources of information that the physician has to
process might include:
(cid:136) Willingness of the patient to go onadiet.
(cid:136) Patient compliance to diet instructions.
(cid:136) Willingness of the patient to accept daily injections for weight loss.
(cid:136) Actual weight loss (from any program).
(cid:136) Actual change in blood sugar.
Each of these are separate flows of information. We can model these by
introducing the set:
I = Thesetofinformationprocessesattimet(thesetmay
t
change as we change strategies, opening up new flows
of information).
We can now express the different flavors of information using
W = The realization of information from source i∈I ,
t+1,i t
W = (W ) .
t+1 t+1,i i∈It
We are going to continue usingWto represent the new information
t+1
arriving, but the reader has to remember that in real applications, it is
typically going to include an entire set of information sources, each with
their own behaviors.
1.7.3 State/decision-dependent processes
There are many applications where the informationWdepends on the
t+1
current stateSand/or the decision x . Some examples include:
t t
(cid:136) A lack of inventory may discourage customers, reducing demand.
(cid:136) Buyingalarge quantity of stocks may increase their prices.
(cid:136) The decision to recommend vaccines may influence the progression
ofadisease.

40 1. Modeling sequential decision problems
(cid:136) Thenumberofpowergeneratorsonlinecanchangeelectricgridprices.
For this reason, it helps to represent the exogenous information asafunc-
tion:
W (S ,x ) = The exogenous information function giving the infor-
t+1 t t
mation arriving in the interval (t,t+1).
Forexample,imaginethatwearebuyingorsellingstockinlargequantities
which may influence the future price. The dynamics might be written as
p =θpp +θpp +θpp +W (S ,x ). (1.26)
t+1 0 t 1 t−1 2 t−2 t+1 t t
The state of this price process would be written
S =(p ,p ,p ).
t t t−1 t−2
The random change in price, given by W (S ,x ), reflects our belief that
t+1 t t
the change in price might depend on the current price (if the price is high,
future changes are likely to be negative) as well as the amount that we are
buying (x >0) or selling (x <0).
t t
Of course, we would like to use historical data to try to separate any
structural influence ofSandxon future prices from the truly exogenous
t t
noise. So, we might proposeamodel
W (S ,x )=θxx +ε ,
t+1 t t t t+1
where we might assume that
ε ∼N(0,|x |σ2),
t+1 t t
Thismodelassumesthatε hasmean 0,andvariancethatgrowswiththe
t+1
absolute value of x . The information W (S ,x ) would then have mean
t t+1 t t
θxx which is positive if we are purchasing shares (x >0), and negative if
t t
we are selling into the market (x <0).
t
This book will continue to useWas the default notation, but the
t+1
reader should beawarethat itmaydependon thecurrent state and/or the
decision made given the state.

1.7. Modeling uncertainty 41
1.7.4 Styles of uncertainty
Identifyingthetypesofinformationisthefirststepinunderstandinguncer-
tainty. The next step is to characterize the different styles of uncertainty.
Asummaryofsomeofthemostimportantwaysthatinformationprocesses
can behave include:
(cid:136) Fine-grained variability – This might arise at time scales of seconds
(even fractions ofasecond), minutes, hours, or daily.
(cid:136) Shifts – The fine-grained variability ofaprocess typically represents
variations aroundamean, but there are times when the mean will
periodically shift toanew level. This might reflect new technology,
competitor adjustments, or changes in the economy.
(cid:136) Bursts and intermittent demands – The spread of disease can create
asurgeofinfectionsasoutbreakscanspreadlocally. Acustomermay
pick upaproduct and recommend it to their friends who then tell
their friends.
(cid:136) Spikes – An incoming snow storm can createajump in the demand
for milk, eggs and toilet paper; a failure ofapower generator can
createaspike in electricity prices.
(cid:136) Spatial events - Weather, diseases and changes in regulations can
create random changes that are regional in nature.
(cid:136) Systemicevents–Theseareeventsthatcanaffectanentirecompany
(spanning international boundaries), an entire country, or even have
a global impact. This might arise because ofacyberattack on com-
munications,changesinpublicperceptions,andnegativeadvertising.
(cid:136) Rare events – Rare events can arise fromanumber of sources such
asearthquakes,diseaseoutbreaks,orterroristattacks. Thesetendto
beeventsthatoccurquiterarely,butwhichcanhaveamajorimpact
on an organization when they do happen.
(cid:136) Contingencies – This category refers to events that might happen,
but for which there is no history. For example, grid operators will
plan forafailure of nuclear power plants. While this may not have
ever happened withinacountry, the grid operator may still want to
prepare for the event if it does happen.

42 1. Modeling sequential decision problems
These behaviors can have an impact on the choice of policy for making
decisions, a topic we deal with next.
Uncertainty is widely recognized asaproblem that companies, organi-
zations and even governments have to plan for. Often overlooked is that
the reason to model uncertainty is to understand how it affects decisions.
Uncertainty is always associated with information processes that arrive in
the future, so we have to think about howadecision made now is affected
by this information in the future.
1.8 Designing policies
A policy isamethod for makingadecision ...any method.
Policies are functions that use the information in the state variable
to makeadecision. This sounds likeawell-defined problem; after all,
the machine learning community is built entirely around the challenge of
finding functions that matcha training data set. However,designingpolicies
is much richer, as evidenced by the diversity of communities that work in
this area.
Figure 1.2 shows the front covers of books representing roughly 15 dis-
tinct fields that all deal with sequential decisions under uncertainty. They
use eight different notational systems, and use fundamentally different ap-
proaches to how they approach modeling. Some confuse policies (which
involve imbedded optimization problems) with objective functions.
1.8.1 Policy performance metrics
Deterministic optimization is characterized by an objective function that
determines whether one decision is better than another. With sequential
decision problems, we will typically have an objective function that eval-
uates the performance ofapolicy, as we did with equations (1.17), (1.21)
and (1.22).
In practice, however, policies are chosen based onanumber of compet-
ing criteria:
(cid:136) Solutionquality-Wearetypicallylookingatperformance(e.g. costs,
profits, health outcomes) over some period of time, as expressed in
the sampled version of the objective in equation (1.22). Since this is
random, we have to consider:

1.8. Designing policies 43
Figure 1.2: Asamplingofmajorbooksrepresentingdifferentfieldsinstochasticopti-
mization.
– Average performance.
– Worst case performance.
(cid:136) Computationalrequirements-Inoperationalsettingsruntimesmat-
ter. As with the objective, the time it takes to computeapolicy is
random, so we need to consider:
– Average execution time.
– Worst case execution times.
(cid:136) Transparency - How easy it is to traceadecision back to input data,
which may have errors.
(cid:136) Flexibility/adaptability - Real-world problems can be complicated,
and we often have to adapt to complex situations.
(cid:136) Methodological complexity - Ifapolicy is being implemented by an
in-houseanalyticsgroup(forexample),theywillhavetoconsiderthe
likelihood that they can getamethod to actually work.
(cid:136) Data requirements - Different policies have different data require-
ments.
Themathematicaloptimizationcommunitiesillustratedinfigure 1.2 might
talk about optimal policies, which implies optimizing the expectation in
equation (1.17). However, it is important to pay attention to all of these
characteristics.

44 1. Modeling sequential decision problems
1.8.2 The four classes of policies
The books in figure 1.2 featuresavariety of ways of making decisions over
time. It turns out that they can all be divided into well-defined classes of
policies. There are two fundamental strategies for creating policies, each of
which can then be further divided into two classes, creating four classes of
policies:
Policy search - This is where you search across methods (functions) for
makingdecisions,simulatingtheirperformance(aswedoinequation
(1.17)), to find the method that works best on average over time.
This may involve searching over different classes of methods, as well
as any tunable parameters foragiven method. This idea opens up
two classes of policies:
1) Policy function approximations (PF As) - These are analytical
functions ofastate that directly specify an action. The order-
up-topolicyinequation(1.7)isagoodexample,alongwithour
policy of using an adjusted forecast in equation (1.15).
2) Cost function approximations (CF As) - These are policies that
involvesolvinganoptimizationproblemthatistypicallyasim-
plification of the original problem, with parameters introduced
to help make the policy work better overtime. Thisisapartic-
ularly powerful idea that is widely used in industry. We have a
number of illustrations of CF Aslaterinthebook(startingwith
chapter 4 to learn the best medication for diabetes).
Lookahead policies -Wecanbuildeffectivepoliciesbyoptimizingacross
thecontribution(orcost)ofadecision, plus an approximation of the
downstream contributions (or cost) resulting from the decision made
now. Again, we can divide these into two more classes of policies:
3) Value function approximations (VF As) - Imagine that we are
traversinganetwork depicted in figure 1.3 where we wish to
findapath from node 1 to node 11. Now imagine that we are
at node S = i = 2, wheretcounts how many links we have
t
traversed. Let V (S ) be the value (assuming we are max-
t+1 t+1
imizing) of the path from node S (such as nodes 4 or 5) to
t+1
node 11 (don't worry about how we obtained V (S )). Let
t+1 t+1

1.8. Designing policies 45
𝑆 (cid:3404) 𝑖(cid:3404)node in network
(cid:3047)
2 8
1 4 6 9
Figure 1.3: Sim𝑉ple(cid:4666)𝑆de(cid:4667)te(cid:3404)rmminaixstic𝐶g(cid:4666)r𝑆ap,h𝑥f(cid:4667)o(cid:3397)rt𝑉raver (cid:4666)si𝑆ngfr(cid:4667)omnode 1 tonode 11.
(cid:3047) (cid:3047) (cid:3051) (cid:3047) (cid:3047)(cid:2878)(cid:2869) (cid:3047)(cid:2878)(cid:2869)
𝑉(cid:3041)(cid:4666)𝑆(cid:3041)(cid:4667)(cid:3404)max 𝐶(cid:4666)𝑆(cid:3041),𝑥(cid:4667)(cid:3397)𝐸 𝑉(cid:3041)(cid:2878)(cid:2869)(cid:4666)𝑆(cid:3041)(cid:2878)(cid:2869)(cid:4667)|𝑆(cid:3041)
(cid:3051) 3
a decisionxbe the link we traverse out of node S = i. The 3
t t
value of being at nodeSwould be given by
t
(cid:0) (cid:1)
V (S )=max C(S ,x )+V (S ) . (1.27)
t t t t t+1 t+1
xt
Equation (1.27) is known as Bellman's equation. When it is
used to find the best path inadeterministic network such as
the one we depicted in figure 1.3, it is fairly easy to visualize.
There are many problems where the transition from stateSto
t
S involves random information that is not known at time t.
t+1
We sawasimple example of randomness in our first inventory
problem in section 1.5.1, andamore complicated example in
section 1.5.2.
Forthesemoregeneralproblems,ifweareinastate S ,makea
t
decisionx ,andthenobservenewinformation W (thatisnot
t t+1
known at time t), it will take us toanew stateSaccording
t+1
to our transition function
S =SM(S ,x ,W ). (1.28)
t+1 t t t+1
Thismeansthatattimetwhenwehavetochoosex ,W isa
t t+1
randomvariable,whichmeansthatSisarandomvariableas
t+1
well. In this case we have to insert an expectation in Bellman's

46 1. Modeling sequential decision problems
equation and write equation (1.27) as
V (S )=max (cid:0) C(S ,x )+E {V (S )|S ,x } (cid:1) . (1.29)
t t
xt
t t Wt+1 t+1 t+1 t t
Here we have inserted the expectation E {·} which literally
Wt+1
means to average over all the random outcomes of W .
t+1
The stochastic version of Bellman's equation in (1.29) is ex-
tremely general. The stateSdoes not just meananode in
t
a graph; it captures any (and all) information relevant to the
problem. The difficulty is that we can no longer compute the
valuefunction V (S ), which in turn means we will not have ac-
t t
cessto V (S )thatweassumedweknewinequations(1.27)
t+1 t+1
and (1.29).
The strategy the research community has used when trying to
apply Bellman's equation is to draw on the field of machine
learningtoestimateastatisticalapproximationthatwearego-
ing to call V (S ). Assuming that we can come up witharea-
t t
sonable approximation V (S ), we would write our policy
t+1 t+1
(our method for makingadecision) using
Xπ(S )=argmax (cid:0) C(S ,x )+E {V (S )|S ,x } (cid:1) . (1.30)
t t t Wt+1 t+1 t+1 t t
xt∈Xt
The notation "argmax f(x)" means the value ofxthat max-
x
imizes the function f(x). The index π carries the information
that specifies the structure of the function f, and any tun-
able parameters θ which we would need in the approximation
V (S ).
t+1 t+1
This class of policy falls under headings such as approximate
dynamicprogrammingand,mostoften,reinforcementlearning.
Whileapowerful idea, it is not easy to apply and depends on
our ability to create an accurate approximation V (S ).
t+1 t+1
There isavery rich literature on methods for approximating
value functions, but it is no panacea. This book will illustrate
this idea inafew places, but readers are cautioned that this
class of policies is quite difficult to use.
4) Directlookaheadapproximations(DL As)-Therearemanyprob-
lems where we simply cannot develop effective policies using
any of the first three classes, and when this happens, we have

1.8. Designing policies 47
to turn to direct lookahead approximations. We will write this
out in its full mathematical form later, but for now, we are
going to describe DL As as makingadecision now while opti-
mizing over a (typically approximate) model that extends over
some planning horizon.
A common DLA is to create an approximate model that is de-
terministic. Thisiswhatwearedoingwhenweuseanavigation
system that finds the shortest path to the destination assuming
that we know the travel time along each link of the network.
Asageneral rule, solving an exact stochastic model of the fu-
ture is almost always impossible, so we are going to investigate
different strategies for approximating the problem.
We illustrated our modeling framework in section 1.5 using two inven-
tory problems, and suggested two simple policies (forms of PF As) with
equations (1.7) and (1.15), but we did this just to haveaconcrete example
ofapolicy. While PF As are widely used in day-to-day decision making,
these are specialized examples.
Bycontrast,wearegoingtoclaimthatthefourclassesofpolicieswejust
outlined (PF As, CF As, VF As and DL As) are universal, in that these cover
any method that we might use to solve any sequential decision problem.
To be clear, these are meta-classes. That is, if we thinkaproblem lends
itself to one particular class, we are not done, since we still have to design
the specific policy within the class. Just the same, we feel that these four
classes providesaroadmap to guide the process for designing policies.
1.8.3 Testing policies
To test the value ofapolicy, we are going to use equation (1.22) which
simulatesapolicy overasingle sample path of the information process
W . The hardest part when simulatingapolicy is typically creating the
t
exogenous information process.
Let ω beasample path, where W (ω),...,W (ω) representsapartic-
1 T
ular sample path. Table 1.3 illustrates 10 sample paths of prices that are
indexed ω1 to ω10. If we choose ω6, then
W (ω6)=44.16.

48 1. Modeling sequential decision problems
t=1 t=2 t=3 t=4 t=5 t=6 t=7 t=8
ωn p p p p p p p p
1 2 3 4 5 6 7 8
ω1 45.00 45.53 47.07 47.56 47.80 48.43 46.93 46.57
ω2 45.00 43.15 42.51 40.51 41.50 41.00 39.16 41.11
ω3 45.00 45.16 45.37 44.30 45.35 47.23 47.35 46.30
ω4 45.00 45.67 46.18 46.22 45.69 44.24 43.77 43.57
ω5 45.00 46.32 46.14 46.53 44.84 45.17 44.92 46.09
ω6 45.00 44.70 43.05 43.77 42.61 44.32 44.16 45.29
ω7 45.00 43.67 43.14 44.78 43.12 42.36 41.60 40.83
ω8 45.00 44.98 44.53 45.42 46.43 47.67 47.68 49.03
ω9 45.00 44.57 45.99 47.38 45.51 46.27 46.02 45.09
ω10 45.00 45.01 46.73 46.08 47.40 49.14 49.03 48.74
Table 1.3: Illustrationofasetofsamplepathsforpricesallstartingat$45.00.
The question is: how do we createasample of observations such as
those depicted in table 1.3? There are three typical strategies:
(cid:136) Createsamplesfromhistoricaldata. Sincethereisonlyoneoutcome
atanypointintime,wecancreatemultiplesamplepathsbycombin-
ing observations from different periods of time. Wemightpickprices
from different years, or demands from different months, or observed
travel times on different days. This approach is not possible when
the exogenous information depends on the stateSor decisions x .
t t
(cid:136) Simulating fromamathematical model. This approach offers the
advantage of being able to generate large samples to get statistically
reliable estimates of the performance ofapolicy. These models can
be very sophisticated, but it is quite easy to create models (even
sophisticated ones) that do not replicate the behavior of real data.
The biggest challenge is capturing correlations, either over time or
between samples (say the demands of different products, the prices
of different stocks, or the wind speed in different locations).
(cid:136) We can test an idea in the field, using observations as they actually
occur. The advantage of this is that we are working with real data
(history may not be the same as the future). The disadvantage is
that it takesaday to observeaday of new data (and we may need
much more thanasingle day of observations).
IfWdependsonthestateSand/ordecisionx ,thenwehavetode-
t+1 t t
vise away to reflect this dependence. Creatingamathematicalmodelmakes

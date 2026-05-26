ГАЗОНОКОСИЛКА.
К сожалению, спонсоры закрыли проект Autonomous Cabbage Robot Platform. 
на текущий момент проект имеет

Текущее состояние:
- HybridAgent
- AStarPlanner
- MissionPlanner
- Predictive Dynamic Avoidance
- Dynamic Traffic Heatmap
- Energy-aware navigation
- True WAIT action
- Benchmark framework
- Auto-tuning system
- Large-map support (50x100 tested)


Зато есть новый проект lawnmower. Хочу адаптировать проект Autonomous Cabbage Robot Platform под новую задачу.
Небольшие изменения в конструкции робота привели к новым данным по потребляемой мощности: 
MOVE_COST = 0.05
TURN_COST = 0.02
CUT_COST = 0.40
но возникают проблемы с навигацией - если с капустой все было просто: 1 клетка - 1 захват капуст, навигация была по меткам на поле, то теперь надо убирать траву так, чтобы не было пробелов по краям клетки (но это можно, наверное решить уменьшением размера клетки так, чтобы она была меньше размера робота и он гарантированно покосит границу клетки). Гораздо более существенная проблема - это не прямоугольные препятствия - круглые, овальные, криволинейные, как решать эту проблему я не знаю. Какие способы навигации существуют по цена/качество? Посмотри на карту тестового участка размером 45х42 метра, заштрихованную область необходимо косить, размер робота 0.5х0.5 метра.
Какие у тебя есть на этот счет соображения?

################################################################################
###### РЕШЕНИЕ ##############
#############################
надо поменять цель миссии:

Было:

найти капусту → доехать → собрать клетку

Должно стать:

покрыть всю mowable area → не оставить дыр → минимизировать повторы → вернуться домой
1. Размер клетки

Робот 0.5 × 0.5 м. Я бы не делал клетку 0.5 м.

Лучше:

CELL_SIZE = 0.25

Тогда карта 45 × 42 м станет примерно:

180 × 168 клеток

Это уже большая карта, но твой проект тестировался на 50×100, а логика large-map уже есть. Придется оптимизировать, но это реалистично.

Почему 0.25 м лучше:

ширина робота = 0.5 м
клетка = 0.25 м
робот перекрывает 2 клетки по ширине

Так мы уменьшаем риск “непрокоса” по краям.

2. Непрямоугольные препятствия

Вот здесь ключевое решение: не пытаться хранить круглые препятствия как прямоугольники.

Надо перейти к occupancy grid:

0 = нельзя ехать / препятствие
1 = можно косить
2 = уже скошено
3 = буфер безопасности

Круглые, овальные и криволинейные объекты просто растеризуются в сетку.

То есть препятствие может быть хоть таким:

клумба
дерево
овальный бассейн
изогнутая дорожка
забор

А для планировщика это будет набор занятых клеток.
3. Обязательный safety buffer

Так как робот 0.5 м, нельзя планировать путь вплотную к препятствию.

Нужно расширять препятствия:

ROBOT_SIZE = 0.5
CELL_SIZE = 0.25
SAFETY_MARGIN = 0.1

inflation_radius_cells = ceil((ROBOT_SIZE / 2 + SAFETY_MARGIN) / CELL_SIZE)

То есть все препятствия “раздуваются” на 1–2 клетки.
А косить можно до границы уже отдельной логикой: центр робота не должен заехать в obstacle, но нож/корпус может покрывать соседние клетки.

4. Лучшие варианты навигации по цена/качество
Вариант A — самый практичный

Grid + A + Boustrophedon coverage*

Цена: низкая
Качество: хорошее
Сложность: умеренная

Это лучший старт для твоего проекта.

Алгоритм:

1. Фото/чертеж участка → binary mask
2. Растеризация в сетку 0.25 м
3. Inflate obstacles
4. Разбить участок на зоны
5. В каждой зоне идти “змейкой”
6. Между зонами ездить через A*
7. Считать энергию: MOVE + TURN + CUT

Твой MissionPlanner уже выбирает сектора по плотности и расстоянию, только вместо cabbages должны быть uncut_cells. Сейчас он строит маршрут по секторам с оставшейся капустой и оценивает сектор через количество целей / расстояние.

Это почти идеально переносится на газон:

cabbages -> uncut grass cells
COLLECT -> CUT / MOW
sector_cabbages -> sector_uncut_cells

Вариант B — лучше для сложных форм

Boustrophedon Cellular Decomposition

Цена: средняя
Качество: очень хорошее
Сложность: выше

Идея: участок автоматически режется на “простые области”, где можно косить змейкой.

Это хорошо для твоей карты, потому что там много ниш, узких проходов и кривых границ.

Вариант C — профессиональный уровень

Coverage planner + polygon map + exact offset paths

Цена: высокая
Качество: отличное
Сложность: высокая

Работает не по клеткам, а по полигонам. Препятствия — настоящие контуры. Строятся параллельные линии покоса с учетом ширины робота.

Это красиво, но для твоего текущего Python-проекта я бы пока не начинал с этого.

5. Что я бы сделал первым

Я бы адаптировал проект так:

CabbageEnv → LawnEnv
grid == 1 → трава, которую надо косить
grid == -1 → препятствие
visited/cut_map → карта скошенной травы
MissionPlanner → выбирает сектор с нескошенной травой
SectorCoveragePlanner → строит змейку по сектору
AStarPlanner → соединяет проходы между зонами
EnergyPredictor → считает MOVE/TURN/CUT
Benchmark → coverage %, overlap %, missed edge cells

Новые метрики:

coverage_rate      # сколько травы скошено
overlap_rate       # сколько лишних повторов
edge_miss_rate     # пропуски у границ
energy_per_m2      # энергия на квадратный метр
turns_per_m2       # насколько маршрут дерганый

6. По твоим новым затратам энергии

Ты написал:

MOVE_COST = 0.05
TURN_COST = 0.02
CUT_COST = 0.40

Это сильно меняет стратегию. Теперь косить дорого, а движение и повороты дешевые.

Значит агент должен:

1. включать нож только на нескошенной траве
2. не косить повторно уже обработанные клетки
3. спокойно делать объезд, если он уменьшает overlap
4. оптимизировать не только путь, но и состояние ножа

В текущем загруженном config.py пока стоят старые значения MOVE_COST = 0.1, TURN_COST = 0.1, CUT_COST = 0.5, их надо заменить.

Главный вывод

Самое выгодное решение сейчас:

Occupancy Grid 0.25 м
+ obstacle inflation
+ секторное покрытие
+ змейка внутри сектора
+ A* между секторами
+ energy-aware knife control

Это даст хороший баланс цена/качество и позволит почти полностью сохранить твою архитектуру. Проект не надо переписывать с нуля — его надо переименовать из “collect all cabbages” в “cover all mowable cells”.

Первый шаг — меняем не агента, а модель мира.

Шаг 1. Новая логика клеток

Вместо:

-1 = obstacle
 0 = empty
 1 = cabbage

делаем:

-1 = obstacle / forbidden
 0 = free but not grass
 1 = uncut grass
 2 = cut grass
 3 = safety buffer
 
Шаг 2 — настоящее coverage planning

Сейчас агент:

ищет ближайшую "капусту"

А газонокосилка должна:

косить полосами
минимизировать overlap
не оставлять дыр
минимизировать повторы

Что будем делать

Мы заменим:

nearest target collection

на:

coverage sweep planning

Как должен работать правильный mower

Идеальная стратегия:

1. выбрать сектор
2. выбрать sweep direction
3. косить полосой
4. в конце полосы сделать разворот
5. перейти на следующую линию
6. повторять

То есть:

██████████ →
← ██████████
██████████ →
← ██████████

Это называется:

Boustrophedon Sweep

(как пашет трактор)

Что это даст

Сразу резко уменьшатся:

overlap_rate
turns
energy_per_m2

А coverage станет почти идеальным.

=== lawn_energy_manager

==== Persistent Sweep State ===
То есть хранить:
	current_lane
	current_direction
	lane_shift_count
	sweep_origin
	visited_lanes
Чтобы после recharge mower продолжал:
	с той же полосы а не начинал почти заново.
=== lawn_lane_memory
=== StripFollower
чтобы он не просто ехал “тупо вперед”, а lane-aware: он будет закрывать текущую полосу, выбирать следующую незавершенную и пытаться перейти к ней.

===== lawn_pygame_renderer ====

=================================
== текущий уровень ====
==================================
Global planner
Sector planner
Coverage route
Energy-aware return
Recovery
Heatmaps
Sector memory
Cut-only return
Benchmark
Visualization

💥💥💥💥 осталось 4 больших слабых места:💥💥💥💥
💥 1. SectorSweepRoute всё ещё слишком тупой #############
	Сейчас:
		фиксированная змейка
	Проблема:
		не учитывает obstacle geometry
		не учитывает узкие проходы
		не учитывает islands

🚀 Boustrophedon Cellular Decomposition ===
	Это реальный алгоритм mower/tractor coverage.
	Идея:
		разбить поле не на квадраты, а на "естественные области"
		между препятствиями
	Тогда:
		каждая область косится идеальной змейкой
		без хаоса вокруг овальных препятствий.
	Это именно то, что используют:
	Husqvarna, Bosch, John Deere, ROS coverage planners

💥 2. A* слишком часто строится заново ################
	Сейчас у тебя:
		очень много replanning
	Следствие:
		горячие зоны
		CPU load

🚀 Hierarchical Path Planning =====
	GLOBAL:
		sector graph
	LOCAL:
		A* внутри сектора
	То есть:
	между секторами = graph routing
	внутри сектора = local coverage

💥 3. Нет “corridor memory” ########################
	Сейчас heatmap только штрафует.

🚀 Persistent Traffic Cost =========
	То есть:
		каждый проход делает corridor дороже
		не только локально, а глобально.
	Тогда исчезнут:
		магистрали
		горячие вертикальные зоны
💥 4. Нет настоящего frontier planning #############
	Сейчас recovery:
		ищет ближайшую траву
🚀 coverage frontier planner =======
		искать frontier boundary между CUT и GRASS
=========================================================	
====================================================
🚜🔥 Boustrophedon Coverage Decomposition ==========
====================================================
	текущие grid-сектора слишком искусственные.
		████
		████
		████
	Из-за этого:
		- agent прыгает между секторами
		- obstacle geometry игнорируется
		- узкие проходы ломают sweep
		- recovery сложный
=== Natural Coverage Cells ===
	   obstacle
	   ██████
	cell A    cell B
	██████    ██████
	██████    ██████
	То есть obstacle автоматически разрезает пространство.
ЭТАП 1 — decomposition
	Делаем:
	free space
		↓
	scanline decomposition
		↓	
	coverage cells
		↓
	adjacency graph
ЭТАП 2 — sweep route per cell
	Каждая natural-cell:
		→→→→
		←←←←
		→→→→
ЭТАП 3 — cell graph planner
	Вместо grid sectors
	будет: coverage graph

=== coverage_traffic_cost ====
	надо сделать маршрут экономнее.
	добавляем штраф за повторные проходы:
	A* не должен любить горячие коридоры

=== order coverage-cells ===
=== knife control ===

==========================
=== Analyze tool PRO 📊 ===
===========================

#############################################

1. RL поверх planner 
	RL НЕ управляет движением напрямую.
	RL выбирает:
		- next coverage cell
		- sweep orientation
		- recharge timing
		- recovery policy
	То есть:
		High-level policy RL
		Low-level deterministic planner
2. adaptive traffic cost
	если overlap растёт → traffic penalty усиливается
	если coverage низкий → traffic penalty ослабляется
3. auto_tune
	CoverageTrafficCost
	CoverageCellOrdering
	LawnEnergyManager
################################################

стратегия:
	Этап A — adaptive traffic cost + auto_tune
	Этап B — RL поверх planner
Почему так:
	1. У нас уже есть стабильный deterministic planner.
	2. Benchmark уже даёт 90–100% success.
	3. Узкое место сейчас — tuning tradeoff:
	   overlap vs energy vs robustness.
	4. Auto-tune быстро даст сильный baseline.
	5. RL без хорошего baseline будет учиться на шумной системе.

RL сейчас привлекательный, но преждевременный. Если запустить RL поверх ещё не оттюненного planner, он будет учиться компенсировать баги и нестабильные веса. Это плохо.

	сначала auto_tune на deterministic planner
		↓
	получить сильные параметры
		↓
	зафиксировать stable baseline
		↓
	потом RL учит high-level decisions

Я бы сделал так:

	1. lawn_auto_tune.py
	2. adaptive_traffic_controller.py
	3. tuned config profile: stable / aggressive / energy_saver
	4. потом RL policy:
	   - choose next cell
	   - choose recharge threshold
	   - choose traffic aggressiveness
	   - choose recovery mode

Итог: adaptive + auto_tune сейчас даст быстрый прирост, а RL потом станет не заменой planner, а “умным диспетчером” поверх уже сильной системы.

======== lawn_auto_tune ====
configs/
    tuned_stable.py
    tuned_aggressive.py
    tuned_energy_saver.py
    tuned_low_overlap.py
	
play_lawn.py или lawn_benchmark_runner.py

		from core.tuning_config import runtime_config

		runtime_config.load_profile(
			"configs.tuned_stable"
		)
lawn_auto_tune.py
		runtime_config.update(cfg)
		
def __repr__(self):
    return str(self.values)		
	
Тогда:
	print(runtime_config)
будет красиво выводить текущий tuning state.

==== adaptive traffic controller ====
=== profile_benchmark_runner ===

=======================================
==== RL OVER PLANNER ==
=========================================
Сделаем RL не вместо planner, а вместо текущего rule-based HighLevelPolicy.act(): он будет выбирать стратегический режим, а движение/маршруты остаются deterministic.
	RL strategic controller
		+
	deterministic planner
		+
	deterministic A*
		+
	deterministic sweep
	
Rule-based HL сейчас — очень сильный teacher policy.
RL пока: почти научился её повторять но ещё не догнал.	

=== imitation learning / policy distillation ===
Вместо: учить RL с нуля мы можем: учить RL копировать rule_hl

Собираем dataset
	Во время обычного rule_hl benchmark сохраняем:
		state -> hl_action
Обучаем RL как classifier:
	state -> action
		через:
	CrossEntropyLoss
Это даст
	RL instantly reaches rule_hl quality
а потом уже можно делать:
	RL fine-tuning over benchmark reward
И это очень правильная robotics architecture
	Rule policy
		↓
	Imitation learning
		↓
	RL fine tuning
Это намного стабильнее, чем pure RL с нуля	
Stage 1 — imitation dataset collection	
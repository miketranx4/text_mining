1. train_finalwriteup.py

This	file	contains	all	the	code	that	is	used	to	pre-process	the	text,	
and	generate	the	matrix	for	the	training	set	for	the	three	models	
(word	features	only,	power	features	only,	and	words	with	power	
features	combined).	The	last	section	of	the	file	marked	by	the	
comment	“Cross	Validation”	has	the	code	for	doing	cross-validation	
on	the	training	set	with	the	random	forest	model	after	we	picked	
the	threshold	from	creating	histogram	(The	code	for	this	part	is	
marked	with	the	comment	“Picking	Threshold”).

2.ROC_curve.py

Note	that	this	code	can	only	be	run	after	the	above	file	
(train_finalwriteup.py)	is	run. This file	contains	code	for	generating	
ROC	curve	as	well	as	the	confusion	matrix.

3. multi_label.py

This	file	contains the	code	for	generating	the	labels	for	the	five	
classes.	After	that,	we	train	random	forest	separately	for	those	five	
classes.	Run	this	file	in	the	same	folder	as	“XtestKaggle2.csv”.	It	will	
make	the	prediction	on	the	Kaggle	dataset,	and	generate	the	
submission	file.

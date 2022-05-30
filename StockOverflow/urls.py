
from django.urls import path
from . import views

urlpatterns =[
    path('',views.index,name='index'),
    path('lstm',views.lstm,name='lstm'),
    path('news',views.news,name='news'),
    path('statistics',views.statistics,name='statistics'),
    path('technical',views.technical,name='technical')
]
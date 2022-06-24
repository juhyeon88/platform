from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_main, name='post_main'),
    path('drw_info/', views.drw_info, name='drw_info'),
    path('drw_country/', views.drw_country, name='drw_country'),
    path('drw_disease/', views.drw_disease, name='drw_disease'),
    path('seq_analyze/', views.seq_analyze, name='seq_analyze'),
    path('seq_analyze_tables/', views.seq_analyze_tables, name='seq_analyze_tables'),
    path('logout/', views.logout, name='logout'),
    path('login/', views.login, name='login'),
    path('signup/', views.signup, name='signup'),
    path('pswdmod/', views.pswdmod, name='pswdmod'),
    path('bert/', views.bert, name='bert'),
    path('about/', views.about, name='about'),
    path('report/', views.report, name='report'),
    path('dengue/', views.dengue, name='dengue'),
    path('dengue_report/', views.dengue_report, name='dengue_report'),
]

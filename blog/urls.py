from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_main, name='post_main'),
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
    path('medisys_about/', views.medisys_about, name='medisys_about'),
    path('medisys_crawl/', views.medisys_crawl, name='medisys_crawl'),
    path('medisys_crawl_func/', views.medisys_crawl_func, name='medisys_crawl_func'),
    path('vaers_about/', views.vaers_about, name='vaers_about'),
    path('vaers_analyze/', views.vaers_analyze, name='vaers_analyze'),
]

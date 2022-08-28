from django.urls import path


from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("pickdate/<int:pk>/<str:s_type>/", views.pickdate, name="pickdate"),
    path("graph/<int:pk>/<str:s_type>/", views.graph, name="graph"),
    path("register", views.register_request, name="register"),
    path("login", views.login_request, name="login"),
    path("logout", views.logout_request, name= "logout"),
]

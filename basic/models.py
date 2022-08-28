from django.db import models

# Create your models here.


class MyStock(models.Model):
    symbol = models.TextField()
    name = models.TextField()
    last_sale = models.TextField()
    net_change = models.TextField()
    p_change = models.TextField()
    market_cap = models.TextField()
    country = models.TextField()
    ipo_year = models.TextField()
    volume = models.TextField()
    sector = models.TextField()
    industry = models.TextField()

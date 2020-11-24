from django.db import models

# Create your models here.
from django.conf import settings
from django.utils import timezone


class Post(models.Model):
    #author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    #title = models.CharField(max_length=200)
    #text = models.TextField()
    location = models.CharField(max_length=200)
    pincode = models.CharField(max_length=6)
    lat = models.CharField(max_length=15)
    long = models.CharField(max_length=15)
    # created_date = models.DateTimeField(default=timezone.now)
    published_date = models.DateTimeField(default=timezone.now)
    # url = models.CharField(max_length=20)
    blog_views = models.IntegerField(default=0)     #,null=True)
    current_view = models.IntegerField(default=0) #, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.location

class FirstVisit(models.Model): #CHECK IF USER VISITS A STORY FOR THE FIRST TIME
    url = models.URLField()
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)

class CurrentViewCheck(models.Model):
    # url = models.URLField()
    storyno = models.CharField(max_length=10)
    user = models.CharField(max_length=50)

# class PostCount(models.Model):
#     total_views = models.IntegerField(default=0)
#     view_total_count = models.ForeignKey(Post, on_delete=models.CASCADE)

#     def __str__(self):
#         return self.total_views
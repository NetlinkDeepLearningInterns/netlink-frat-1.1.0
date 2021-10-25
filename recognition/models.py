from django.db import models

class Camera(models.Model):
    ip=models.CharField(max_length=64)
    username=models.CharField(max_length=255)
    password=models.CharField(max_length=255)
    channel=models.CharField(max_length=255)
    location=models.CharField(max_length=255)
    cameratype=models.CharField(max_length=20)
    # 'rtsp://mdpadmin:admin@10.95.9.27:554/Streaming/Channels/101/'
    def getUrl(self):
        url='rtsp://'+self.username+':'+self.password+'@'+self.ip+'/Streaming/Channels/'+self.channel+'/'
        return url


# Create your models here.

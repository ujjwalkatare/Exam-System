from django.contrib import admin
from .models import*
# Register your models here.
admin.site.register(Question)
admin.site.register(Exam)
admin.site.register(Response)
admin.site.register(Block)
admin.site.register(Profile)
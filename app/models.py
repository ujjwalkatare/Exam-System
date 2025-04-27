from django.db import models
from django.contrib.auth.models import User
from django.db import models

class Exam(models.Model):
    title = models.CharField(max_length=200)
    date = models.DateField()
    duration = models.IntegerField()  # Duration in minutes
    total_marks = models.IntegerField()
    txn_hash = models.CharField(max_length=66, blank=True, null=True)  # Add this line

    def __str__(self):
        return self.title

class Question(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='questions')
    text = models.CharField(max_length=500)
    option1 = models.CharField(max_length=200)
    option2 = models.CharField(max_length=200)
    option3 = models.CharField(max_length=200)
    option4 = models.CharField(max_length=200)
    correct_answer = models.CharField(max_length=200)
    txn_hash = models.CharField(max_length=66, blank=True, null=True)  # Add this line

    def __str__(self):
        return self.text


class Response(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    answer = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.student} - {self.exam.title} - {self.question.text}"
    
import hashlib
from django.db import models

class Block(models.Model):
    index = models.PositiveIntegerField()  # Block index
    timestamp = models.DateTimeField(auto_now_add=True)
    data = models.TextField()  # Data (Exam, Question, Response)
    hash = models.CharField(max_length=64)  # SHA-256 hash of the block
    previous_hash = models.CharField(max_length=64)  # Previous block's hash

    def __str__(self):
        return f"Block {self.index} - {self.timestamp}"

    def save(self, *args, **kwargs):
        if not self.hash:
            self.hash = self.calculate_hash()
        super().save(*args, **kwargs)

    def calculate_hash(self):
        data_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}"
        return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

    @classmethod
    def create_block(cls, data, previous_block):
        index = previous_block.index + 1 if previous_block else 0
        previous_hash = previous_block.hash if previous_block else '0'
        block = Block(index=index, data=data, previous_hash=previous_hash)
        block.save()
        return block


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=[('student', 'Student'), ('teacher', 'Teacher')])

    def __str__(self):
        return self.user.username

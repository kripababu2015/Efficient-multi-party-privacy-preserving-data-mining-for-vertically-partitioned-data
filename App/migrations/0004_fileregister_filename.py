# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2020-06-29 12:43
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('App', '0003_fileregister'),
    ]

    operations = [
        migrations.AddField(
            model_name='fileregister',
            name='filename',
            field=models.CharField(default='MCAData.csv', max_length=200),
        ),
    ]

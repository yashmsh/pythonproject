# Generated by Django 3.2.12 on 2022-02-27 02:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_auto_20220227_0212'),
    ]

    operations = [
        migrations.AddField(
            model_name='datamain',
            name='percentage',
            field=models.IntegerField(default=100),
            preserve_default=False,
        ),
    ]

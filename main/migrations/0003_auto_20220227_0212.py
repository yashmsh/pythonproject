# Generated by Django 3.2.12 on 2022-02-27 02:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_datamain_checkedimage'),
    ]

    operations = [
        migrations.AddField(
            model_name='datamain',
            name='Alignment',
            field=models.CharField(default='it works', max_length=255),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='datamain',
            name='Colour',
            field=models.CharField(default='colour is correct', max_length=255),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='datamain',
            name='Rx',
            field=models.CharField(default='Rx is correct', max_length=255),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='datamain',
            name='Spelling',
            field=models.CharField(default='Spelling is correct', max_length=255),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='datamain',
            name='checkedImage',
            field=models.CharField(max_length=255),
        ),
    ]

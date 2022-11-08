# Lightfield

## Initials and create sub-aperture views

This task is to convert the lightfield to 5d numpy arrays for lateral processing and creating mosaic sub-aperture views for visualizing lightfield.

```bash
python initialize.py
```

## Refocusing and focal-stack simulation

The aperature has two shape of choice: circle of square, and different focus point.

```
python refocus.py
```

## All-in-focus image and depth from focus

```
python all-in-focus.py
```

## Focus aperture stack and confocal stereo

```
python confocal.py
```

## Unstructured lightfield

```
python unstructured_lightfield.py
```

import os
import time
import shutil
import subprocess
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shutterspeed", "-s", type=str, default="1/100")
    parser.add_argument("--iso", "-i", type=str, default="100")
    parser.add_argument("--fnumber", "-f", type=str, default="5.6")
    parser.add_argument(
        "--whitebalance",
        "-w",
        type=int,
        default=4,
        help="0 Automatic, 1 Daylight, 2 Fluorescent,3 Tungsten, 4 Flash, 5 Cloudy, 6 Shade, 7 Preset",
    )
    parser.add_argument("--output", "-o", type=str, default="test")
    parser.add_argument("--out_size", type=str, default="640")
    parser.add_argument("--count", "-c", type=int, default=10)
    parser.add_argument("--interval", "-t", type=int, default=1)

    args = parser.parse_args()

    if os.path.exists(args.output + ".nef"):
        os.remove(args.output + ".nef")

    if os.path.exists(args.output + ".jpg"):
        os.remove(args.output + ".jpg")

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    os.makedirs(args.output)

    for i in range(args.count):
        time.sleep(args.interval)
        subprocess.run(["gphoto2", "--auto-detect"])
        subprocess.run(
            [
                "gphoto2",
                "--set-config",
                "/main/capturesettings/shutterspeed={}".format(args.shutterspeed),
            ]
        )

        subprocess.run(
            ["gphoto2", "--set-config", "/main/imgsettings/iso={}".format(args.iso)]
        )
        subprocess.run(
            [
                "gphoto2",
                "--set-config",
                "/main/capturesettings/f-number={}".format(args.fnumber),
            ]
        )
        subprocess.run(
            [
                "gphoto2",
                "--set-config",
                "/main/imgsettings/whitebalance={}".format(args.whitebalance),
            ]
        )

        # subprocess.run(
        #     ["gphoto2", "--get-config", "/main/capturesettings/shutterspeed"]
        # )
        # subprocess.run(["gphoto2", "--get-config", "/main/imgsettings/iso"])
        # subprocess.run(["gphoto2", "--get-config", "/main/capturesettings/f-number"])

        subprocess.run(
            [
                "gphoto2",
                "--capture-image-and-download",
                "--filename",
                os.path.join(args.output, "{:04}".format(i) + ".%C"),
            ]
        )
        # develop and get linear tif image
        subprocess.run(
            [
                "dcraw",
                "-q",
                "3",
                "-T",
                "-4",
                "-f",
                "-w,",
                "-o",
                "1",
                os.path.join(args.output, "{:04}".format(i) + ".nef"),
            ]
        )

        # resize
        subprocess.run(
            [
                "convert",
                os.path.join(args.output, "{:04}".format(i) + ".tiff"),
                "-resize",
                args.out_size + "x-1",
                os.path.join(args.output, "{:04}".format(i) + ".tif"),
            ]
        )
        # delete tiff
        os.remove(os.path.join(args.output, "{:04}".format(i) + ".tiff"))

        subprocess.run(["gphoto2", "--reset"])

/****************************************************************************
 * Copyright ©2017 Brian Curless.  All rights reserved.  Permission is hereby
 * granted to students registered for University of Washington CSE 457 or CSE
 * 557 for use solely during Autumn Quarter 2017 for purposes of the course.
 * No other use, copying, distribution, or modification is permitted without
 * prior written consent. Copyrights for third-party components of this work
 * must be honored.  Instructors interested in reusing these course materials
 * should contact the author.
 ****************************************************************************/
#include <mainwindow.h>
#include <QApplication>
#include <QSurfaceFormat>

int main(int argc, char *argv[])
{
    // Represents the format of a renderable surface
    // NOTE: If not set as default, Mac renders as black screen
    QSurfaceFormat glFormat;
    glFormat.setRenderableType( QSurfaceFormat::OpenGL );
    glFormat.setMajorVersion( 4 );
    glFormat.setMinorVersion( 0 );
    glFormat.setProfile( QSurfaceFormat::CoreProfile ); // Functionality deprecated in OpenGL 3.0 is not available.
    glFormat.setSwapBehavior( QSurfaceFormat::DoubleBuffer );
    glFormat.setStencilBufferSize( 8 );
    glFormat.setSamples(4);

    QSurfaceFormat::setDefaultFormat(glFormat);
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
#ifdef __APPLE__
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif
    QApplication application( argc, argv );

    MainWindow window;
    if (application.arguments().contains("--run-tests")) {

        QString render_folder = "assets/trace/tests_output/";

        if (application.arguments().contains("--render-folder")){
            int render_folder_index = application.arguments().indexOf("--render-folder") + 1;
            render_folder = application.arguments().at(render_folder_index);
        }

        QFile file("assets/trace/tests.txt");
        if(!file.open(QIODevice::ReadOnly)) {
            qInfo() << "Text file could not open.";
            return 0;
        }

        QTextStream in(&file);

        while(!in.atEnd()) {
            QString line = in.readLine();
            QString scene_filename = "assets/trace/tests/" + line + ".yaml";
            QString render_filename = render_folder + "/" + line + ".png";
            window.NonInteractiveRender(scene_filename, render_filename);
        }

        file.close();

        return 0;
    } else {
        window.show();
        return application.exec();
    }
}

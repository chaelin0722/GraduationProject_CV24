package com.example.myapplication;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

public class reminderBroadcast  extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        /*
        NotificationCompat.Builder builder = new NotificationCompat.Builder(context,"notifyLemubit")
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setContentTitle("remind remuit academy")
                .setContentText("this is a reminder...")
                .setPriority(NotificationCompat.PRIORITY_DEFAULT);

        NotificationManagerCompat notificationManager = NotificationManagerCompat.from(context);

        notificationManager.notify(200, builder.build());
*/
        NotificationCompat.Builder builder = new NotificationCompat.Builder(context,"notifyUDP")
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setContentTitle("udp info")
                .setContentText("info")
                .setPriority(NotificationCompat.PRIORITY_DEFAULT);

        NotificationManagerCompat notificationManager = NotificationManagerCompat.from(context);
        notificationManager.notify(200, builder.build());

    }
}

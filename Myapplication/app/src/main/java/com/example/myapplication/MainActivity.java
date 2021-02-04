package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;
import android.util.Log;
import android.app.AlarmManager;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.widget.TextView;
import android.os.Message;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;

public class MainActivity extends AppCompatActivity {
/*
    private final Handler mHandler = new Handler();
    ReceiveData rcvServer;
    String info;
    String rcvIp, rcvPort, rcvPacket;
    public static final String TAG = "CH : ";
*/
    TextView packetView, rcvIpText, rcvPortText;
    String rcvIp, rcvPort, rcvPacket;
    String ind;

    private final Handler mHandler = new Handler();

    //데이터를 송신받을떄 실행할 코드
    private final Runnable rnb = new Runnable() {
        @Override
        public void run() {
            //레이아웃에 존재하는 UI를 불러옴
            packetView = findViewById(R.id.packetView);
            rcvIpText = findViewById(R.id.ipView);
            rcvPortText = findViewById(R.id.portView);
            //UI에 UDP 수신 하면서 얻은 정보를 업데이트 함
            //packetView.setText(rcvPacket);
            packetView.setText(ind);
            rcvIpText.setText(rcvIp);
            rcvPortText.setText(rcvPort);


        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //createNotificationChannel();

        Button button = findViewById(R.id.button);
        button.setOnClickListener(V ->{
            Toast.makeText(this, "server set",Toast.LENGTH_SHORT).show();


//UDP 수신 쓰레드 설정 및 시작부분
            //
            ReceiveData rcvServer = new ReceiveData(9090);
            rcvServer.start();

        });
    } //end oncreate

    class ReceiveData extends Thread {
        Handler handler = mHandler;
        DatagramSocket rSocket;

        //UDP 수신 포트를 열기(원하는 port 아무거나)
        public ReceiveData(int port){
            try {
                //수신 소켓 생성
                rSocket = new DatagramSocket(port);
            } catch (SocketException e) {
                e.printStackTrace();
            }
        }

        public void run(){
            try {
                while (true){
                    //받을 패킷 생성
                    byte[] buf = new byte[1024];

                    //패킷으로 변경 (바이트 버퍼, 버퍼길이)
                    DatagramPacket packet = new DatagramPacket(buf, buf.length);

                    //데이터 수신 대기
                    rSocket.receive(packet);

                    //클라이언트에서 보낸 메시지 저장!
                    ind = new String(packet.getData(),0,packet.getLength());
                    setNotification(ind);


                    //패킷을 보낸 상대방의 ip를 저장함
                    InetAddress ina = packet.getAddress();
                    rcvIp = ina.toString();

                    //패킷을 보낸 상대방의 port를 저장함
                    int inp = packet.getPort();
                    rcvPort = String.valueOf(inp);

                    //수신받은 데이터를 문자열로 변환
                    rcvPacket = new String(buf);

                    //에코하기 위해 송신용 패킷을 만듦 (받은 패킷, 패킷 길이, 상대방 IP, 상대방 PORT)
                    packet = new DatagramPacket(buf, buf.length, ina, inp);

                    //이 기기로 UDP송신을 한 상대방에게 받은패킷을 돌려줌
                    rSocket.send(packet);

                    //데이터를 받았다면 UI로 표현해주기 위해 runnable 사용
                    handler.post(rnb);

                    //쓰레드를 인터럽트로 종료시키기 위해 sleep을 사용함
                    sleep(20);
                }
            }catch (InterruptedException e){
                rcvPacket = e.toString();
                handler.post(rnb);
            }
            catch (Exception e){
                rcvPacket = e.toString();
                handler.post(rnb);
            }
        }
    }

    private void setNotification(CharSequence message) {
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, "notifyUDP")
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setContentTitle(message)
                .setContentText("not working?")
                .setPriority(NotificationCompat.PRIORITY_DEFAULT);

        NotificationManagerCompat notificationManager = NotificationManagerCompat.from(this);
        notificationManager.notify(200, builder.build());


        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {

            CharSequence name = "notification Channel";
            String description = "for over oreo";
            int importance = NotificationManager.IMPORTANCE_DEFAULT;
            NotificationChannel channel = new NotificationChannel("notifyUDP", name, importance);
            channel.setDescription(description);

            assert notificationManager != null;
            notificationManager.createNotificationChannel(channel);
        }else builder.setSmallIcon(R.mipmap.ic_launcher);

        assert notificationManager != null;
        notificationManager.notify(1234, builder.build());
    }

}